"""
Classification pipeline for proteomics data.

Loads seen.csv and unseen.csv from preprocess.py, then trains and evaluates
three classifiers (Logistic Regression, Random Forest, XGBoost) on a feature
subset loaded from an external CSV.

Hyperparameter tuning uses Monte Carlo Cross-Validation (MCCV): the seen
data is randomly split into balanced train/val sets ``--n-splits`` times,
and each candidate parameter combination is scored across all splits.
The best parameters are then used to refit on all of seen.csv before
evaluating on the held-out unseen (test) set.

Feature file contract
---------------------
The --features-path CSV must be a single-column file with header "protein".
Values must match column names in seen.csv / unseen.csv exactly
(e.g., seq.1234.56). Any upstream feature selection method can produce
this file as long as it follows this format.

Outputs (when --save-results):
  results/<results-subdir>/classification_results.csv
  results/<results-subdir>/cv_results_<model>.csv   (per-split breakdown)
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import logging
import time

import pandas as pd

from src.core.classifier_utils import (
    MonteCarloCV,
    build_lr_pipeline,
    build_rf_pipeline,
    build_xgb_pipeline,
    evaluate_classifier,
    train_classifier,
)
from src.core.data_utils import load_data, load_features
from src.core.logging_utils import setup_logging


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start = time.time()

    parser = argparse.ArgumentParser(
        description="Train and evaluate classifiers on selected proteomics features."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/seen.csv"),
        help="Path to the seen (train+val) CSV (default: data/processed/seen.csv).",
    )
    parser.add_argument(
        "--unseen-path",
        type=Path,
        default=Path("data/processed/unseen.csv"),
        help="Path to the unseen (test) CSV (default: data/processed/unseen.csv).",
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        required=True,
        help=(
            "Path to selected features CSV (single column, header 'protein'). "
            "E.g. results/ttest/selected_features_k10.csv"
        ),
    )
    parser.add_argument(
        "--results-subdir",
        type=str,
        default="classifier",
        help="Subdirectory under results/ for output files (default: classifier).",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save outputs to disk and log to file; otherwise log to terminal only.",
    )
    parser.add_argument(
        "--log-subdir",
        type=str,
        default="classifier",
        help="Subdirectory under logs/ for log files (default: classifier).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.20,
        help="Fraction of seen data used as the balanced validation set (default: 0.20).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=50,
        help="Number of Monte Carlo CV splits for hyperparameter tuning (default: 50).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for MCCV and reproducibility (default: 42).",
    )
    args = parser.parse_args()

    logger = setup_logging(args.save_results, args.log_subdir, "classifier")

    logger.info("Starting classifier.py")
    logger.info(
        "Args: data_path=%s  unseen_path=%s  features_path=%s  "
        "val_frac=%s  n_splits=%d  random_state=%s  save_results=%s",
        args.data_path, args.unseen_path, args.features_path,
        args.val_frac, args.n_splits, args.random_state, args.save_results,
    )

    # ------------------------------------------------------------------
    # Step 1: Load data and features
    # ------------------------------------------------------------------
    seen = load_data(args.data_path, logger)
    unseen = load_data(args.unseen_path, logger)
    features = load_features(args.features_path, logger)

    X = seen[features]
    y = seen["ards"].astype(int)
    X_test = unseen[features]
    y_test = unseen["ards"].astype(int)

    logger.info("Feature matrices — X: %s  X_test: %s", X.shape, X_test.shape)

    # ------------------------------------------------------------------
    # Step 2: Prepare CV splitter and model definitions
    # ------------------------------------------------------------------
    cv = MonteCarloCV(
        n_splits=args.n_splits,
        val_frac=args.val_frac,
        random_state=args.random_state,
    )
    logger.info(
        "MonteCarloCV: n_splits=%d  val_frac=%.2f  random_state=%d",
        args.n_splits, args.val_frac, args.random_state,
    )

    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    spw = (neg / pos) if pos > 0 else 1.0

    models = [
        ("LogisticRegression", *build_lr_pipeline()),
        ("RandomForest", *build_rf_pipeline()),
        ("XGBoost", *build_xgb_pipeline(scale_pos_weight=spw)),
    ]

    # ------------------------------------------------------------------
    # Step 3: Train and evaluate each model
    # ------------------------------------------------------------------
    results_dir = Path("results") / args.results_subdir

    all_rows: list[dict] = []

    for name, pipeline, param_grid in models:
        fitted, cv_df, summary = train_classifier(
            X, y, pipeline, param_grid, cv, logger, model_name=name,
        )

        test_metrics = evaluate_classifier(
            fitted, X_test, y_test, logger, model_name=name,
        )

        # CV summary row
        all_rows.append({
            "model": name,
            "split": "cv",
            "auc": summary["auc_mean"],
            "auc_std": summary["auc_std"],
            "accuracy": summary["accuracy_mean"],
            "accuracy_std": summary["accuracy_std"],
            "f1": summary["f1_mean"],
            "f1_std": summary["f1_std"],
        })

        # Test row
        all_rows.append({
            "model": name,
            "split": "test",
            "auc": test_metrics["auc"],
            "auc_std": None,
            "accuracy": test_metrics["accuracy"],
            "accuracy_std": None,
            "f1": test_metrics["f1"],
            "f1_std": None,
        })

        # Save per-model CV breakdown
        if args.save_results:
            results_dir.mkdir(parents=True, exist_ok=True)
            cv_out = results_dir / f"cv_results_{name}.csv"
            cv_df.to_csv(cv_out, index=False)
            logger.info("Saved %s CV breakdown to: %s", name, cv_out)

    results_df = pd.DataFrame(all_rows)
    logger.info("Classification summary:\n%s", results_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Step 4: Save combined results
    # ------------------------------------------------------------------
    if args.save_results:
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / "classification_results.csv"
        results_df.to_csv(out_path, index=False)
        logger.info("Saved classification results to: %s", out_path)

    logger.info("Finished in %.2f s", time.time() - start)


if __name__ == "__main__":
    main()
