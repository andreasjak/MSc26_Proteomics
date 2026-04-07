"""Validate protein sub-selection strategies on unseen data.

This script trains a chosen classifier on seen data using three strategies:
1) Random feature subset
2) Top N proteins from each cluster (by pi)
3) Top K proteins globally by pi

It compares unseen performance and then runs a permutation test on unseen labels
for the best strategy (selected by unseen F1 score).

Inputs:
- Cluster/protein DataFrame (CSV path or DataFrame) with columns: protein, cluster, pi
- Seen data (CSV path or DataFrame)
- Unseen data (CSV path or DataFrame)
- Classifier choice (rf or lr)

Outputs:
- Results DataFrame with unseen metrics
- Comparison plot
- Permutation plot and permutation summary for the best model
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc as pr_auc,
    confusion_matrix,
    recall_score,
    precision_score,
    balanced_accuracy_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.core.logging_utils import setup_logging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_df(input_data: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(input_data, pd.DataFrame):
        return input_data.copy()
    return pd.read_csv(input_data)


def _build_model(classifier: str, random_state: int) -> Pipeline:
    if classifier == "lr":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
            ]
        )
    if classifier == "rf":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=100, random_state=random_state)),
            ]
        )
    raise ValueError("classifier must be 'rf' or 'lr'")


def _validate_cluster_df(cluster_df: pd.DataFrame) -> None:
    required_cols = {"protein", "cluster", "pi"}
    missing = required_cols - set(cluster_df.columns)
    if missing:
        raise ValueError(f"cluster_df is missing required columns: {sorted(missing)}")


def _prepare_strategies(
    cluster_df: pd.DataFrame,
    seen_cols: list[str],
    top_per_cluster: int,
    top_k_pi: int,
    random_state: int,
) -> dict[str, list[str]]:
    """Build feature lists for each strategy."""
    cluster_df = cluster_df.copy()
    cluster_df["protein"] = cluster_df["protein"].astype(str)

    # Keep only proteins available in seen/unseen matrices.
    cluster_df = cluster_df[cluster_df["protein"].isin(seen_cols)].copy()
    if cluster_df.empty:
        raise ValueError("No proteins from cluster_df found in seen/unseen data columns.")

    top2_per_cluster = (
        cluster_df.sort_values(["cluster", "pi"], ascending=[True, False])
        .groupby("cluster", as_index=False)
        .head(top_per_cluster)["protein"]
        .drop_duplicates()
        .tolist()
    )

    top10_pi = (
        cluster_df.sort_values("pi", ascending=False)["protein"]
        .drop_duplicates()
        .head(top_k_pi)
        .tolist()
    )

    rng = np.random.default_rng(random_state)
    n_random = max(1, len(top2_per_cluster))
    random_feats = rng.choice(np.array(seen_cols), size=min(n_random, len(seen_cols)), replace=False).tolist()

    return {
        "Random": random_feats,
        f"Top{top_per_cluster}PerCluster": top2_per_cluster,
        f"Top{top_k_pi}Pi": top10_pi,
    }


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_on_unseen(
    seen_df: pd.DataFrame,
    unseen_df: pd.DataFrame,
    strategies: dict[str, list[str]],
    classifier: str = "rf",
    label_col: str = "ards",
    protein_prefix: str = "seq.",
    random_state: int = 42,
) -> pd.DataFrame:
    """Train on seen and evaluate on unseen with imbalanced-data metrics."""
    all_proteins = [c for c in seen_df.columns if str(c).startswith(protein_prefix)]
    common_proteins = [c for c in all_proteins if c in unseen_df.columns]

    X_train = seen_df[common_proteins].copy()
    y_train = seen_df[label_col].astype(int).copy()

    X_test = unseen_df[common_proteins].copy()
    y_test = unseen_df[label_col].astype(int).copy()

    rows: list[dict[str, Any]] = []
    for label, features in strategies.items():
        features = [f for f in features if f in X_train.columns and f in X_test.columns]
        if len(features) == 0:
            continue

        model = _build_model(classifier=classifier, random_state=random_state)
        model.fit(X_train[features].values, y_train)

        y_proba = model.predict_proba(X_test[features].values)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        # Compute AUC-PR (main metric for imbalanced data)
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        auc_pr = pr_auc(recall, precision)

        # Other imbalanced-friendly metrics
        sensitivity = recall_score(y_test, y_pred, zero_division=0)  # recall on positive class
        specificity = recall_score(y_test, y_pred, pos_label=0, zero_division=0)  # recall on negative
        precision_val = precision_score(y_test, y_pred, zero_division=0)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        rows.append(
            {
                "label": label,
                "n_features": len(features),
                "features": features,
                "auc_pr": float(auc_pr),
                "auc_roc": float(roc_auc_score(y_test, y_proba)),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "precision": float(precision_val),
                "balanced_accuracy": float(balanced_acc),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "confusion_matrix": cm,
                "y_test": y_test.values,
                "y_pred": y_pred,
            }
        )

    if not rows:
        raise ValueError("No valid strategies to evaluate after feature filtering.")

    return (
        pd.DataFrame(rows)
        .sort_values(["auc_pr", "auc_roc"], ascending=[False, False])
        .reset_index(drop=True)
    )


def permutation_test_best_model_unseen(
    seen_df: pd.DataFrame,
    unseen_df: pd.DataFrame,
    best_label: str,
    best_features: list[str],
    classifier: str = "rf",
    label_col: str = "ards",
    n_permutations: int = 200,
    random_state: int = 42,
) -> dict[str, Any]:
    """Permutation test using AUC-PR on unseen labels."""
    X_train = seen_df[best_features].copy()
    y_train = seen_df[label_col].astype(int).copy()

    X_test = unseen_df[best_features].copy()
    y_test = unseen_df[label_col].astype(int).copy()

    model = _build_model(classifier=classifier, random_state=random_state)
    model.fit(X_train.values, y_train)

    y_proba = model.predict_proba(X_test.values)[:, 1]

    # Compute real AUC-PR
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    real_auc_pr = float(pr_auc(recall, precision))

    y_test_arr = np.asarray(y_test)
    perm_auc_prs = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y_test_arr)
        prec, rec, _ = precision_recall_curve(y_perm, y_proba)
        perm_auc_prs.append(pr_auc(rec, prec))

    perm_auc_prs = np.array(perm_auc_prs)
    p_value = float((perm_auc_prs >= real_auc_pr).mean())

    return {
        "best_label": best_label,
        "n_features": len(best_features),
        "real_auc_pr_unseen": real_auc_pr,
        "perm_auc_pr_mean": float(perm_auc_prs.mean()),
        "p_value": p_value,
        "perm_auc_prs": perm_auc_prs,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_confusion_matrices(results_df: pd.DataFrame, out_path: Path) -> None:
    """Plot confusion matrices for all strategies."""
    n_strategies = len(results_df)
    fig, axes = plt.subplots(1, n_strategies, figsize=(5 * n_strategies, 4))
    if n_strategies == 1:
        axes = [axes]

    for idx, (_, row) in enumerate(results_df.iterrows()):
        cm = row["confusion_matrix"]
        label = row["label"]

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[idx],
            cbar=False,
            xticklabels=["Non-ARDS", "ARDS"],
            yticklabels=["Non-ARDS", "ARDS"],
        )
        axes[idx].set_title(f"{label}\n(n={row['n_features']})", fontsize=10)
        axes[idx].set_ylabel("True label")
        axes[idx].set_xlabel("Predicted label")

    fig.suptitle("Confusion matrices for all strategies", fontsize=13)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_comparison(results_df: pd.DataFrame, out_path: Path) -> None:
    """Plot all metrics for strategy comparison (imbalanced data metrics)."""
    metrics = ["auc_pr", "auc_roc", "sensitivity", "specificity", "precision", "balanced_accuracy"]
    labels = results_df["label"].tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    colors = ["#2ecc71", "#3498db", "#95a5a6"][: len(labels)]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = results_df[metric].values
        bars = ax.barh(labels, values, color=colors, alpha=0.85)
        ax.set_xlim(0, 1)
        ax.set_xlabel(metric)
        ax.set_title(metric)

        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center")

    fig.suptitle("Sub-selection strategy comparison (imbalanced-data metrics)", fontsize=14)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_permutation(perm_result: dict[str, Any], out_path: Path) -> None:
    """Plot permutation test results for AUC-PR."""
    perm_aucs = perm_result["perm_auc_prs"]
    real_auc = perm_result["real_auc_pr_unseen"]
    p_value = perm_result["p_value"]
    best_label = perm_result["best_label"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(perm_aucs, bins=40, alpha=0.7, color="steelblue", label="Permuted unseen labels")
    ax.axvline(real_auc, color="red", linewidth=2, label=f"Real unseen AUC-PR = {real_auc:.3f}")
    ax.axvline(np.percentile(perm_aucs, 95), color="orange", linestyle="--", label="95th percentile")
    ax.set_title(f"Permutation test (AUC-PR, best model, unseen)\n{best_label}\np-value = {p_value:.4f}")
    ax.set_xlabel("AUC-PR")
    ax.legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_sub_selection_validation(
    cluster_input: pd.DataFrame | str | Path,
    seen_input: pd.DataFrame | str | Path,
    unseen_input: pd.DataFrame | str | Path,
    classifier: str = "rf",
    label_col: str = "ards",
    protein_prefix: str = "seq.",
    top_per_cluster: int = 2,
    top_k_pi: int = 10,
    n_permutations: int = 200,
    random_state: int = 42,
    comparison_plot_out: Path | None = None,
    confusion_matrix_plot_out: Path | None = None,
    permutation_plot_out: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run full validation workflow and return (results_df, permutation_result)."""
    cluster_df = _load_df(cluster_input)
    seen_df = _load_df(seen_input)
    unseen_df = _load_df(unseen_input)

    _validate_cluster_df(cluster_df)

    seen_proteins = [c for c in seen_df.columns if str(c).startswith(protein_prefix)]
    strategies = _prepare_strategies(
        cluster_df=cluster_df,
        seen_cols=seen_proteins,
        top_per_cluster=top_per_cluster,
        top_k_pi=top_k_pi,
        random_state=random_state,
    )

    results_df = evaluate_on_unseen(
        seen_df=seen_df,
        unseen_df=unseen_df,
        strategies=strategies,
        classifier=classifier,
        label_col=label_col,
        protein_prefix=protein_prefix,
        random_state=random_state,
    )

    # Select the best strategy by AUC-PR; use AUC-ROC as tie-break.
    best_row = (
        results_df.sort_values(["auc_pr", "auc_roc"], ascending=[False, False])
        .iloc[0]
    )
    best_label = str(best_row["label"])
    best_features = list(best_row["features"])

    perm_result = permutation_test_best_model_unseen(
        seen_df=seen_df,
        unseen_df=unseen_df,
        best_label=best_label,
        best_features=best_features,
        classifier=classifier,
        label_col=label_col,
        n_permutations=n_permutations,
        random_state=random_state,
    )

    perm_result["best_auc_pr_unseen"] = float(best_row["auc_pr"])
    perm_result["best_auc_roc_unseen"] = float(best_row["auc_roc"])

    if comparison_plot_out is not None:
        plot_comparison(results_df, comparison_plot_out)
    if confusion_matrix_plot_out is not None:
        plot_confusion_matrices(results_df, confusion_matrix_plot_out)
    if permutation_plot_out is not None:
        plot_permutation(perm_result, permutation_plot_out)

    return results_df, perm_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    start = time.time()

    parser = argparse.ArgumentParser(
        description="Validate sub-selection strategies on unseen data and run permutation test on best model."
    )
    parser.add_argument("--cluster-path", type=Path, required=True, help="Path to cluster dataframe CSV.")
    parser.add_argument("--seen-path", type=Path, default=Path("data/processed/seen.csv"), help="Seen CSV path.")
    parser.add_argument("--unseen-path", type=Path, default=Path("data/processed/unseen.csv"), help="Unseen CSV path.")
    parser.add_argument("--classifier", type=str, default="rf", choices=["rf", "lr"], help="Classifier type.")
    parser.add_argument("--label-col", type=str, default="ards", help="Label column.")
    parser.add_argument("--protein-prefix", type=str, default="seq.", help="Protein column prefix.")
    parser.add_argument("--top-per-cluster", type=int, default=2, help="Top proteins per cluster.")
    parser.add_argument("--top-k-pi", type=int, default=10, help="Top proteins by pi.")
    parser.add_argument("--n-permutations", type=int, default=200, help="Number of permutations.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/sub_selection_validation"),
        help="Output directory for results and plots.",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="If set, save logs to logs/sub_selection_validation/. Otherwise log to terminal.",
    )
    args = parser.parse_args()

    logger = setup_logging(
        save_results=args.save_results,
        log_subdir="sub_selection_validation",
        script_name="sub_selection_validation",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    comparison_plot_out = args.out_dir / "comparison_unseen.png"
    confusion_matrix_plot_out = args.out_dir / "confusion_matrices_unseen.png"
    permutation_plot_out = args.out_dir / "permutation_best_unseen.png"

    results_df, perm_result = run_sub_selection_validation(
        cluster_input=args.cluster_path,
        seen_input=args.seen_path,
        unseen_input=args.unseen_path,
        classifier=args.classifier,
        label_col=args.label_col,
        protein_prefix=args.protein_prefix,
        top_per_cluster=args.top_per_cluster,
        top_k_pi=args.top_k_pi,
        n_permutations=args.n_permutations,
        random_state=args.random_state,
        comparison_plot_out=comparison_plot_out,
        confusion_matrix_plot_out=confusion_matrix_plot_out,
        permutation_plot_out=permutation_plot_out,
    )

    results_csv = args.out_dir / "strategy_results_unseen.csv"
    results_df.to_csv(results_csv, index=False)

    perm_json = args.out_dir / "permutation_best_unseen.json"
    perm_to_save = {
        "best_label": perm_result["best_label"],
        "n_features": perm_result["n_features"],
        "best_auc_pr_unseen": perm_result["best_auc_pr_unseen"],
        "best_auc_roc_unseen": perm_result["best_auc_roc_unseen"],
        "real_auc_pr_unseen": perm_result["real_auc_pr_unseen"],
        "perm_auc_pr_mean": perm_result["perm_auc_pr_mean"],
        "p_value": perm_result["p_value"],
    }
    with open(perm_json, "w", encoding="utf-8") as f:
        json.dump(perm_to_save, f, indent=2)

    logger.info("Saved results: %s", results_csv)
    logger.info("Saved comparison plot: %s", comparison_plot_out)
    logger.info("Saved confusion matrices: %s", confusion_matrix_plot_out)
    logger.info("Saved permutation plot: %s", permutation_plot_out)
    logger.info("Saved permutation summary: %s", perm_json)
    logger.info(
        "Best strategy: %s | unseen AUC-PR=%.3f | unseen AUC-ROC=%.3f | p-value=%.4f",
        perm_result["best_label"],
        perm_result["best_auc_pr_unseen"],
        perm_result["best_auc_roc_unseen"],
        perm_result["p_value"],
    )
    logger.info("Done in %.2f s", time.time() - start)


if __name__ == "__main__":
    main()
