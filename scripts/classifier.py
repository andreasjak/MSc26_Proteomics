"""
Classification pipeline for proteomics data.

Loads seen.csv and unseen.csv from preprocess.py, splits seen.csv into
a balanced train/val set, and trains three classifiers (Logistic Regression,
Random Forest, XGBoost) on a feature subset loaded from an external CSV.

Feature file contract
---------------------
The --features-path CSV must be a single-column file with header "protein".
Values must match column names in seen.csv / unseen.csv exactly
(e.g., seq.1234.56). Any upstream feature selection method can produce
this file as long as it follows this format.

Outputs (when --save-results):
  results/<results-subdir>/classification_results.csv
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Hyperparameter grids (identical to original ttests.py)
# ---------------------------------------------------------------------------
LR_PARAM_GRID = {"clf__C": [10.0, 15, 18, 21]}

RF_PARAM_GRID = {
    "clf__n_estimators": [300, 600],
    "clf__max_depth": [None, 5, 10],
    "clf__min_samples_split": [2, 5, 8, 10],
    "clf__min_samples_leaf": [2, 5],
    "clf__max_features": ["sqrt", 0.5],
}

# XGBoost grid is built inside train_and_evaluate() because
# scale_pos_weight depends on the training class distribution.


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(
    save_results: bool, log_subdir: str, script_name: str
) -> logging.Logger:
    """
    Configure logging:
      - save_results=False → StreamHandler (terminal) only
      - save_results=True  → FileHandler (file) only, no terminal output
    Log path: logs/<log_subdir>/<script_name>_YYYYMMDD_HHMMSS.log
    """
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if save_results:
        log_dir = Path("logs") / log_subdir
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"{script_name}_{timestamp}.log"
        handler: logging.Handler = logging.FileHandler(log_path)
    else:
        handler = logging.StreamHandler()

    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(
    data_path: Path,
    unseen_path: Path,
    features_path: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Load seen.csv, unseen.csv, and the feature list.

    Validates:
      1. Every feature name exists in both seen and unseen columns.
      2. No rows overlap between seen and unseen.
    """
    logger.info("Loading seen data from %s", data_path)
    seen = pd.read_csv(data_path)
    logger.info("Seen shape: %s", seen.shape)

    logger.info("Loading unseen data from %s", unseen_path)
    unseen = pd.read_csv(unseen_path)
    logger.info("Unseen shape: %s", unseen.shape)

    logger.info("Loading feature list from %s", features_path)
    features_df = pd.read_csv(features_path)
    features = features_df["protein"].tolist()
    logger.info("Features loaded: %d", len(features))

    ## --- Validate feature names ------------------------------------------
    #missing_seen = [f for f in features if f not in seen.columns]
    #missing_unseen = [f for f in features if f not in unseen.columns]
    #if missing_seen or missing_unseen:
    #    msg_parts = []
    #    if missing_seen:
    #        msg_parts.append(f"Missing in seen: {missing_seen}")
    #    if missing_unseen:
    #        msg_parts.append(f"Missing in unseen: {missing_unseen}")
    #    raise ValueError(
    #        "Feature names not found in data columns. " + " | ".join(msg_parts)
    #    )

    ## --- Check for row overlap between seen and unseen --------------------
    #shared_cols = [c for c in seen.columns if c in unseen.columns]
    #merged = seen[shared_cols].merge(unseen[shared_cols], on=shared_cols, how="inner")
    #if len(merged) > 0:
    #    logger.warning(
    #        "Found %d overlapping row(s) between seen and unseen! "
    #        "This may indicate a data leakage issue in preprocess.py.",
    #        len(merged),
    #    )
    #else:
    #    logger.info("Overlap check passed — no shared rows between seen and unseen.")

    return seen, unseen, features


# ---------------------------------------------------------------------------
# Balanced train/val split
# ---------------------------------------------------------------------------

def balanced_train_val_split(
    df: pd.DataFrame,
    val_frac: float,
    random_state: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a balanced validation set with equal ARDS / non-ARDS counts.
    Identical logic to the original ttests.py split.
    """
    df = df.copy()

    df_pos = df[df["ards"] == 1]
    df_neg = df[df["ards"] == 0]

    n_val_total = int(round(len(df) * val_frac))
    n_each = min(n_val_total // 2, len(df_pos), len(df_neg))

    val_pos = df_pos.sample(n=n_each, random_state=random_state)
    val_neg = df_neg.sample(n=n_each, random_state=random_state)

    val_df = pd.concat([val_pos, val_neg]).sample(
        frac=1, random_state=random_state
    )

    train_df = df.drop(val_df.index)

    logger.info(
        "Balanced split — train: %d (%d ARDS / %d non-ARDS)  |  "
        "val: %d (%d ARDS / %d non-ARDS)",
        len(train_df),
        int((train_df["ards"] == 1).sum()),
        int((train_df["ards"] == 0).sum()),
        len(val_df),
        int((val_df["ards"] == 1).sum()),
        int((val_df["ards"] == 0).sum()),
    )
    return train_df, val_df


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Train LR, RF, and XGBoost with GridSearchCV (PredefinedSplit),
    evaluate on val and test, and return a summary DataFrame.

    Returns DataFrame with columns: model, split, auc, accuracy.
    """
    # ------------------------------------------------------------------
    # Combine train + val for PredefinedSplit grid search
    # ------------------------------------------------------------------
    X_tv = pd.concat([X_train, X_val], axis=0)
    y_tv = pd.concat([y_train, y_val], axis=0)

    test_fold = np.r_[
        -np.ones(len(X_train), dtype=int),
        np.zeros(len(X_val), dtype=int),
    ]
    ps = PredefinedSplit(test_fold=test_fold)
    logger.info("Combined train+val for grid search: %s", X_tv.shape)

    rows: list[dict] = []

    # ==================================================================
    # Logistic Regression
    # ==================================================================
    logger.info("Training Logistic Regression...")

    pipe_lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=10000, solver="lbfgs", class_weight="balanced",
        )),
    ])

    grid_lr = GridSearchCV(
        pipe_lr,
        param_grid=LR_PARAM_GRID,
        cv=ps,
        scoring="roc_auc",
        refit=False,
        n_jobs=-1,
    )
    grid_lr.fit(X_tv, y_tv)

    logger.info("LR best params: %s", grid_lr.best_params_)
    logger.info("LR VAL AUC (grid score): %.4f", grid_lr.best_score_)

    best_lr = pipe_lr.set_params(**grid_lr.best_params_)
    best_lr.fit(X_train, y_train)

    proba_val = best_lr.predict_proba(X_val)[:, 1]
    pred_val = best_lr.predict(X_val)

    auc_val = roc_auc_score(y_val, proba_val)
    acc_val = accuracy_score(y_val, pred_val)
    logger.info("LR VAL — AUC: %.4f  Accuracy: %.4f", auc_val, acc_val)
    logger.info("LR VAL confusion matrix:\n%s", confusion_matrix(y_val, pred_val))
    logger.info("LR VAL classification report:\n%s",
                classification_report(y_val, pred_val, digits=3))
    rows.append({"model": "LogisticRegression", "split": "val",
                 "auc": auc_val, "accuracy": acc_val})

    # Test — refit on train+val
    final_lr = pipe_lr.set_params(**grid_lr.best_params_)
    final_lr.fit(X_tv, y_tv)

    proba_test = final_lr.predict_proba(X_test)[:, 1]
    pred_test = final_lr.predict(X_test)

    auc_test = roc_auc_score(y_test, proba_test)
    acc_test = accuracy_score(y_test, pred_test)
    logger.info("LR TEST — AUC: %.4f  Accuracy: %.4f", auc_test, acc_test)
    logger.info("LR TEST confusion matrix:\n%s", confusion_matrix(y_test, pred_test))
    logger.info("LR TEST classification report:\n%s",
                classification_report(y_test, pred_test, digits=3))
    rows.append({"model": "LogisticRegression", "split": "test",
                 "auc": auc_test, "accuracy": acc_test})

    # ==================================================================
    # Random Forest
    # ==================================================================
    logger.info("Training Random Forest...")

    pipe_rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )),
    ])

    grid_rf = GridSearchCV(
        pipe_rf,
        param_grid=RF_PARAM_GRID,
        cv=ps,
        scoring="roc_auc",
        refit=False,
        n_jobs=-1,
    )
    grid_rf.fit(X_tv, y_tv)

    logger.info("RF best params: %s", grid_rf.best_params_)
    logger.info("RF VAL AUC (grid score): %.4f", grid_rf.best_score_)

    best_rf = pipe_rf.set_params(**grid_rf.best_params_)
    best_rf.fit(X_train, y_train)

    proba_val = best_rf.predict_proba(X_val)[:, 1]
    pred_val = best_rf.predict(X_val)

    auc_val = roc_auc_score(y_val, proba_val)
    acc_val = accuracy_score(y_val, pred_val)
    logger.info("RF VAL — AUC: %.4f  Accuracy: %.4f", auc_val, acc_val)
    logger.info("RF VAL confusion matrix:\n%s", confusion_matrix(y_val, pred_val))
    logger.info("RF VAL classification report:\n%s",
                classification_report(y_val, pred_val, digits=3))
    rows.append({"model": "RandomForest", "split": "val",
                 "auc": auc_val, "accuracy": acc_val})

    # Test — refit on train+val
    final_rf = pipe_rf.set_params(**grid_rf.best_params_)
    final_rf.fit(X_tv, y_tv)

    proba_test = final_rf.predict_proba(X_test)[:, 1]
    pred_test = final_rf.predict(X_test)

    auc_test = roc_auc_score(y_test, proba_test)
    acc_test = accuracy_score(y_test, pred_test)
    logger.info("RF TEST — AUC: %.4f  Accuracy: %.4f", auc_test, acc_test)
    logger.info("RF TEST confusion matrix:\n%s", confusion_matrix(y_test, pred_test))
    logger.info("RF TEST classification report:\n%s",
                classification_report(y_test, pred_test, digits=3))
    rows.append({"model": "RandomForest", "split": "test",
                 "auc": auc_test, "accuracy": acc_test})

    # ==================================================================
    # XGBoost
    # ==================================================================
    logger.info("Training XGBoost...")

    pipe_xgb = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )),
    ])

    # scale_pos_weight depends on training class distribution
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = (neg / pos) if pos > 0 else 1.0

    xgb_param_grid = {
        "clf__n_estimators": [300, 600],
        "clf__learning_rate": [0.03, 0.1],
        "clf__max_depth": [2, 3, 4],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
        "clf__reg_lambda": [1.0, 10.0],
        "clf__min_child_weight": [1, 3, 5, 10],
        "clf__scale_pos_weight": [1.0, spw],
    }

    grid_xgb = GridSearchCV(
        pipe_xgb,
        param_grid=xgb_param_grid,
        cv=ps,
        scoring="roc_auc",
        refit=False,
        n_jobs=-1,
    )
    grid_xgb.fit(X_tv, y_tv)

    logger.info("XGB best params: %s", grid_xgb.best_params_)
    logger.info("XGB VAL AUC (grid score): %.4f", grid_xgb.best_score_)

    best_xgb = pipe_xgb.set_params(**grid_xgb.best_params_)
    best_xgb.fit(X_train, y_train)

    proba_val = best_xgb.predict_proba(X_val)[:, 1]
    pred_val = (proba_val >= 0.5).astype(int)

    auc_val = roc_auc_score(y_val, proba_val)
    acc_val = accuracy_score(y_val, pred_val)
    logger.info("XGB VAL — AUC: %.4f  Accuracy: %.4f", auc_val, acc_val)
    logger.info("XGB VAL confusion matrix:\n%s", confusion_matrix(y_val, pred_val))
    logger.info("XGB VAL classification report:\n%s",
                classification_report(y_val, pred_val, digits=3))
    rows.append({"model": "XGBoost", "split": "val",
                 "auc": auc_val, "accuracy": acc_val})

    # Test — refit on train+val
    final_xgb = pipe_xgb.set_params(**grid_xgb.best_params_)
    final_xgb.fit(X_tv, y_tv)

    proba_test = final_xgb.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)

    auc_test = roc_auc_score(y_test, proba_test)
    acc_test = accuracy_score(y_test, pred_test)
    logger.info("XGB TEST — AUC: %.4f  Accuracy: %.4f", auc_test, acc_test)
    logger.info("XGB TEST confusion matrix:\n%s", confusion_matrix(y_test, pred_test))
    logger.info("XGB TEST classification report:\n%s",
                classification_report(y_test, pred_test, digits=3))
    rows.append({"model": "XGBoost", "split": "test",
                 "auc": auc_test, "accuracy": acc_test})

    return pd.DataFrame(rows)


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
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the balanced train/val split (default: 42).",
    )
    args = parser.parse_args()

    logger = setup_logging(args.save_results, args.log_subdir, "classifier")

    logger.info("Starting classifier.py")
    logger.info(
        "Args: data_path=%s  unseen_path=%s  features_path=%s  "
        "val_frac=%s  random_state=%s  save_results=%s",
        args.data_path, args.unseen_path, args.features_path,
        args.val_frac, args.random_state, args.save_results,
    )

    # Step 1: Load data and validate
    seen, unseen, features = load_data(
        args.data_path, args.unseen_path, args.features_path, logger
    )

    # Step 2: Balanced train/val split
    train_df, val_df = balanced_train_val_split(
        seen, args.val_frac, args.random_state, logger
    )

    # Step 3: Subset to selected features
    X_train = train_df[features]
    y_train = train_df["ards"].astype(int)
    X_val = val_df[features]
    y_val = val_df["ards"].astype(int)
    X_test = unseen[features]
    y_test = unseen["ards"].astype(int)

    logger.info(
        "Feature matrices — X_train: %s  X_val: %s  X_test: %s",
        X_train.shape, X_val.shape, X_test.shape,
    )

    # Step 4: Train and evaluate all models
    results_df = train_and_evaluate(
        X_train, y_train, X_val, y_val, X_test, y_test, logger
    )

    logger.info("Classification summary:\n%s", results_df.to_string(index=False))

    # Step 5: Save results
    if args.save_results:
        results_dir = Path("results") / args.results_subdir
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / "classification_results.csv"
        results_df.to_csv(out_path, index=False)
        logger.info("Saved classification results to: %s", out_path)

    logger.info("Finished in %.2f s", time.time() - start)


if __name__ == "__main__":
    main()
