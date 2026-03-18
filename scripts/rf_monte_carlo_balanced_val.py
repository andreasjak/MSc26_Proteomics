#!/usr/bin/env python3
"""
Random Forest with Monte Carlo hyperparameter search and balanced validation sets.

Use cases
---------
1) Start from filtered_data.csv:
   - create seen/unseen split
   - do Monte Carlo tuning inside seen with balanced validation sets
   - fit final model on seen
   - evaluate on unseen

2) Start from existing seen.csv and unseen.csv:
   - skip outer split
   - do the same tuning/evaluation

Validation strategy
-------------------
For each Monte Carlo repetition:
- sample a fraction of ARDS cases into validation
- sample the same number of non-ARDS cases into validation
- all remaining samples go into training

This gives a balanced validation set, while final evaluation is done on the
held-out unseen set with its natural class imbalance.

Outputs
-------
Saves to results_dir:
- summary.csv
- all_runs.csv
- best_params.json
- unseen_metrics.json
- unseen_predictions.csv
- rf_feature_importances.csv
- final_model.joblib
- outer_split_info.json   (if split created from filtered_data.csv)
- seen.csv / unseen.csv   (if split created from filtered_data.csv)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def split_seen_unseen(
    df: pd.DataFrame,
    target_col: str = "ards",
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Outer stratified seen/unseen split."""
    seen, unseen = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col],
    )
    return seen.reset_index(drop=True), unseen.reset_index(drop=True)


def split_X_y_proteins_only(
    df: pd.DataFrame,
    target_col: str = "ards",
    protein_prefix: str = "seq.",
) -> tuple[pd.DataFrame, pd.Series]:
    """Use only protein columns as features."""
    protein_cols = [c for c in df.columns if str(c).startswith(protein_prefix)]
    if not protein_cols:
        raise ValueError(
            f"No protein columns found with prefix '{protein_prefix}'. "
            f"First columns were: {list(df.columns[:10])}"
        )

    X = df[protein_cols].copy()
    y = df[target_col].astype(int).copy()
    return X, y


def make_balanced_val_split(
    seen_df: pd.DataFrame,
    target_col: str = "ards",
    val_frac_ards: float = 0.30,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create one Monte Carlo split with balanced validation set.

    - choose n_val_pos = round(val_frac_ards * number_of_ards)
    - choose equally many negatives for validation
    - all remaining samples go to training
    """
    rng = np.random.default_rng(random_state)

    df_pos = seen_df[seen_df[target_col] == 1].copy()
    df_neg = seen_df[seen_df[target_col] == 0].copy()

    n_pos = len(df_pos)
    n_neg = len(df_neg)

    if n_pos < 2:
        raise ValueError("Too few ARDS samples to create balanced validation sets.")

    n_val_pos = max(1, int(round(val_frac_ards * n_pos)))
    n_val_pos = min(n_val_pos, n_pos - 1)  # leave at least one positive for train
    n_val_neg = n_val_pos

    if n_neg <= n_val_neg:
        raise ValueError("Too few non-ARDS samples to create balanced validation sets.")

    pos_idx = rng.choice(df_pos.index.to_numpy(), size=n_val_pos, replace=False)
    neg_idx = rng.choice(df_neg.index.to_numpy(), size=n_val_neg, replace=False)

    val_idx = np.concatenate([pos_idx, neg_idx])

    val_df = (
        seen_df.loc[val_idx]
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
    train_df = seen_df.drop(index=val_idx).reset_index(drop=True)

    return train_df, val_df


def evaluate_binary_metrics(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute binary classification metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall_ards": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_ards": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_ards": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(specificity),
        "prevalence_ards": float(y_true.mean()),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    return metrics


def build_pipeline(params: dict, random_state: int) -> Pipeline:
    """Construct Random Forest pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
            bootstrap=True,
            **params,
        )),
    ])


def default_param_grid(grid_size: str = "small") -> dict:
    """Reasonable RF grids for p >> n proteomics."""
    if grid_size == "small":
        return {
            "n_estimators": [300, 600, 2000],
            "max_depth": [None, 20],
            "min_samples_split": [2, 10],
            "min_samples_leaf": [1, 3],
            "max_features": ["sqrt", 0.1],
        }
    if grid_size == "medium":
        return {
            "n_estimators": [300, 600, 1000, 2000],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 10],
            "min_samples_leaf": [1, 3, 5, 8, 15],
            "max_features": ["sqrt", 0.1, 0.2],
        }
    raise ValueError("grid_size must be 'small' or 'medium'")


def monte_carlo_rf_search_balanced_val(
    seen_df: pd.DataFrame,
    target_col: str = "ards",
    n_splits: int = 20,
    val_frac_ards: float = 0.30,
    random_state: int = 42,
    scoring_metric: str = "pr_auc",
    protein_prefix: str = "seq.",
    param_grid: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Monte Carlo tuning over balanced validation splits.
    """
    if param_grid is None:
        param_grid = default_param_grid("small")

    all_runs = []
    grid_list = list(ParameterGrid(param_grid))

    print(f"Hyperparameter combinations: {len(grid_list)}")
    print(f"Monte Carlo splits: {n_splits}")
    print(f"Total model fits: {len(grid_list) * n_splits}")

    for split_id in range(n_splits):
        split_seed = random_state + split_id

        train_df, val_df = make_balanced_val_split(
            seen_df=seen_df,
            target_col=target_col,
            val_frac_ards=val_frac_ards,
            random_state=split_seed,
        )

        X_train, y_train = split_X_y_proteins_only(
            train_df, target_col=target_col, protein_prefix=protein_prefix
        )
        X_val, y_val = split_X_y_proteins_only(
            val_df, target_col=target_col, protein_prefix=protein_prefix
        )

        for params in grid_list:
            pipe = build_pipeline(params=params, random_state=split_seed)
            pipe.fit(X_train, y_train)
            y_val_proba = pipe.predict_proba(X_val)[:, 1]

            metrics = evaluate_binary_metrics(y_val, y_val_proba, threshold=0.5)

            row = {
                "split_id": split_id,
                "split_seed": split_seed,
                "n_train": int(len(train_df)),
                "n_val": int(len(val_df)),
                "n_ards_train": int(train_df[target_col].sum()),
                "n_ards_val": int(val_df[target_col].sum()),
                **params,
                **metrics,
            }
            all_runs.append(row)

        print(
            f"[{split_id + 1}/{n_splits}] "
            f"done - train={len(train_df)}, val={len(val_df)}, "
            f"ARDS in val={int(val_df[target_col].sum())}"
        )

    all_runs_df = pd.DataFrame(all_runs)

    group_cols = [
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
    ]

    summary_df = (
        all_runs_df
        .groupby(group_cols, dropna=False)
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            mean_pr_auc=("pr_auc", "mean"),
            std_pr_auc=("pr_auc", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            mean_recall_ards=("recall_ards", "mean"),
            std_recall_ards=("recall_ards", "std"),
            mean_precision_ards=("precision_ards", "mean"),
            std_precision_ards=("precision_ards", "std"),
            mean_f1_ards=("f1_ards", "mean"),
            std_f1_ards=("f1_ards", "std"),
            mean_specificity=("specificity", "mean"),
            std_specificity=("specificity", "std"),
        )
        .reset_index()
    )

    metric_col = f"mean_{scoring_metric}"
    if metric_col not in summary_df.columns:
        raise ValueError(
            f"scoring_metric='{scoring_metric}' not found. "
            f"Choose from pr_auc, roc_auc, balanced_accuracy, recall_ards, "
            f"precision_ards, f1_ards, specificity."
        )

    summary_df = summary_df.sort_values(metric_col, ascending=False).reset_index(drop=True)
    return summary_df, all_runs_df


def row_to_best_params(best_row: pd.Series) -> dict:
    """Convert summary row to RF params dict."""
    max_depth = best_row["max_depth"]
    if pd.isna(max_depth):
        max_depth = None
    else:
        max_depth = int(max_depth)

    return {
        "n_estimators": int(best_row["n_estimators"]),
        "max_depth": max_depth,
        "min_samples_split": int(best_row["min_samples_split"]),
        "min_samples_leaf": int(best_row["min_samples_leaf"]),
        "max_features": best_row["max_features"],
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Random Forest Monte Carlo tuning with balanced validation sets."
    )

    # Input mode
    parser.add_argument(
        "--mode",
        choices=["filtered", "split"],
        default="filtered",
        help="filtered: start from filtered_data.csv, split: start from seen.csv + unseen.csv",
    )
    parser.add_argument(
        "--filtered-path",
        type=Path,
        default=Path("data/processed/filtered_data.csv"),
        help="Path to filtered_data.csv when mode=filtered",
    )
    parser.add_argument(
        "--seen-path",
        type=Path,
        default=Path("data/processed/seen.csv"),
        help="Path to seen.csv when mode=split",
    )
    parser.add_argument(
        "--unseen-path",
        type=Path,
        default=Path("data/processed/unseen.csv"),
        help="Path to unseen.csv when mode=split",
    )

    # Split / target / features
    parser.add_argument("--target-col", type=str, default="ards")
    parser.add_argument("--protein-prefix", type=str, default="seq.")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)

    # Monte Carlo search
    parser.add_argument("--n-splits", type=int, default=20)
    parser.add_argument("--val-frac-ards", type=float, default=0.30)
    parser.add_argument(
        "--scoring-metric",
        type=str,
        default="f1_ards",
        choices=[
            "pr_auc",
            "roc_auc",
            "balanced_accuracy",
            "recall_ards",
            "precision_ards",
            "f1_ards",
            "specificity",
        ],
    )
    parser.add_argument(
        "--grid-size",
        type=str,
        default="small",
        choices=["small", "medium"],
        help="small is much faster; medium is more exhaustive",
    )

    # Output
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/random_forest/monte_carlo_balanced_val"),
    )

    args = parser.parse_args()
    t0 = time.time()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("Starting script with args:")
    print(vars(args))

    # -------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------
    if args.mode == "filtered":
        df = pd.read_csv(args.filtered_path)
        seen, unseen = split_seen_unseen(
            df=df,
            target_col=args.target_col,
            test_size=args.test_size,
            random_state=args.random_state,
        )

        # Save actual split for reproducibility / later SHAP analysis
        seen.to_csv(args.results_dir / "seen.csv", index=False)
        unseen.to_csv(args.results_dir / "unseen.csv", index=False)

        split_info = {
            "mode": "filtered",
            "filtered_path": str(args.filtered_path),
            "test_size": args.test_size,
            "random_state": args.random_state,
            "n_total": int(len(df)),
            "n_seen": int(len(seen)),
            "n_unseen": int(len(unseen)),
            "n_ards_total": int(df[args.target_col].sum()),
            "n_ards_seen": int(seen[args.target_col].sum()),
            "n_ards_unseen": int(unseen[args.target_col].sum()),
        }
        with open(args.results_dir / "outer_split_info.json", "w") as f:
            json.dump(split_info, f, indent=2)

    else:
        seen = pd.read_csv(args.seen_path)
        unseen = pd.read_csv(args.unseen_path)

    print(f"Seen shape:   {seen.shape}, ARDS={int(seen[args.target_col].sum())}")
    print(f"Unseen shape: {unseen.shape}, ARDS={int(unseen[args.target_col].sum())}")

    X_seen, y_seen = split_X_y_proteins_only(
        seen, target_col=args.target_col, protein_prefix=args.protein_prefix
    )
    X_unseen, y_unseen = split_X_y_proteins_only(
        unseen, target_col=args.target_col, protein_prefix=args.protein_prefix
    )

    print(f"Protein features used: {X_seen.shape[1]}")

    # -------------------------------------------------------------
    # Monte Carlo tuning
    # -------------------------------------------------------------
    param_grid = default_param_grid(args.grid_size)

    summary_df, all_runs_df = monte_carlo_rf_search_balanced_val(
        seen_df=seen,
        target_col=args.target_col,
        n_splits=args.n_splits,
        val_frac_ards=args.val_frac_ards,
        random_state=args.random_state,
        scoring_metric=args.scoring_metric,
        protein_prefix=args.protein_prefix,
        param_grid=param_grid,
    )

    summary_df.to_csv(args.results_dir / "summary.csv", index=False)
    all_runs_df.to_csv(args.results_dir / "all_runs.csv", index=False)

    print("\nTop hyperparameter settings:")
    print(summary_df.head(10).to_string(index=False))

    # -------------------------------------------------------------
    # Final model on full seen
    # -------------------------------------------------------------
    best_row = summary_df.iloc[0]
    best_params = row_to_best_params(best_row)

    with open(args.results_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    print("\nBest params:")
    print(best_params)

    final_model = build_pipeline(params=best_params, random_state=args.random_state)
    final_model.fit(X_seen, y_seen)

    # Save trained model
    joblib.dump(final_model, args.results_dir / "final_model.joblib")
    print(f"Saved final model to {args.results_dir / 'final_model.joblib'}")

    # -------------------------------------------------------------
    # Evaluate on unseen
    # -------------------------------------------------------------
    y_unseen_proba = final_model.predict_proba(X_unseen)[:, 1]
    y_unseen_pred = (y_unseen_proba >= 0.5).astype(int)

    unseen_metrics = evaluate_binary_metrics(y_unseen, y_unseen_proba, threshold=0.5)
    unseen_metrics["best_params"] = best_params

    unseen_predictions_df = pd.DataFrame({
        "y_true": y_unseen.astype(int).values,
        "y_proba": y_unseen_proba,
        "y_pred": y_unseen_pred,
    }, index=X_unseen.index)

    unseen_predictions_df.to_csv(args.results_dir / "unseen_predictions.csv", index=False)

    with open(args.results_dir / "unseen_metrics.json", "w") as f:
        json.dump(unseen_metrics, f, indent=2)

    print("\nUnseen metrics:")
    print(json.dumps(unseen_metrics, indent=2))

    # -------------------------------------------------------------
    # Feature importances
    # -------------------------------------------------------------
    rf_model = final_model.named_steps["rf"]
    rf_importance_df = pd.DataFrame({
        "Protein": X_seen.columns,
        "rf_importance": rf_model.feature_importances_,
    }).sort_values("rf_importance", ascending=False)

    rf_importance_df.to_csv(args.results_dir / "rf_feature_importances.csv", index=False)

    print("\nTop 20 RF proteins:")
    print(rf_importance_df.head(20).to_string(index=False))

    print(f"\nFinished in {(time.time() - t0) / 60:.2f} minutes")


if __name__ == "__main__":
    main()