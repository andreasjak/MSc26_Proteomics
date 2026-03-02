"""
classifier_utils.py
-------------------
Reusable classification utilities for the MSc26 proteomics project.

Provides:
- ``MonteCarloCV`` — sklearn-compatible CV splitter that produces balanced
  validation folds (equal ARDS / non-ARDS counts) via repeated random
  sampling.
- Pipeline factory functions for Logistic Regression, Random Forest, and
  XGBoost (each returns a ``(Pipeline, param_grid)`` pair).
- ``train_classifier`` — hyperparameter tuning with ``GridSearchCV`` using
  any sklearn-compatible CV splitter, then refit on all training data.
- ``evaluate_classifier`` — evaluation of a fitted pipeline on a held-out
  test set.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Monte Carlo Cross-Validation splitter
# ---------------------------------------------------------------------------

class MonteCarloCV:
    """Sklearn-compatible CV splitter with balanced validation folds.

    Each split randomly samples a validation set with equal numbers of
    positive and negative samples, using the remainder as the training set.

    Parameters
    ----------
    n_splits : int, optional
        Number of random train/val splits (default: 50).
    val_frac : float, optional
        Approximate fraction of the full dataset to use as validation
        (default: 0.20).  The actual size is rounded to achieve equal
        class counts.
    random_state : int, optional
        Base random seed.  Split *i* uses ``random_state + i`` for
        reproducibility (default: 42).
    """

    def __init__(
        self,
        n_splits: int = 50,
        val_frac: float = 0.20,
        random_state: int = 42,
    ) -> None:
        self.n_splits = n_splits
        self.val_frac = val_frac
        self.random_state = random_state

    # -- sklearn CV interface ---------------------------------------------

    def split(self, X, y=None, groups=None):
        """Yield (train_indices, val_indices) for each split.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Required — class labels used to balance the validation set.
        groups : ignored
        """
        if y is None:
            raise ValueError("MonteCarloCV requires y for balanced splits.")

        y_arr = np.asarray(y)
        pos_idx = np.where(y_arr == 1)[0]
        neg_idx = np.where(y_arr == 0)[0]

        n_total = len(y_arr)
        n_val_total = int(round(n_total * self.val_frac))
        n_each = min(n_val_total // 2, len(pos_idx), len(neg_idx))

        for i in range(self.n_splits):
            rng = np.random.RandomState(self.random_state + i)
            val_pos = rng.choice(pos_idx, size=n_each, replace=False)
            val_neg = rng.choice(neg_idx, size=n_each, replace=False)
            val_indices = np.concatenate([val_pos, val_neg])
            train_indices = np.setdiff1d(np.arange(n_total), val_indices)
            yield train_indices, val_indices

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splitting iterations."""
        return self.n_splits


# ---------------------------------------------------------------------------
# Pipeline factories
# ---------------------------------------------------------------------------

def build_lr_pipeline() -> tuple[Pipeline, dict]:
    """Return a Logistic Regression pipeline and its hyperparameter grid.

    Returns
    -------
    pipeline : Pipeline
        ``SimpleImputer`` → ``StandardScaler`` → ``LogisticRegression``.
    param_grid : dict
        Grid for ``GridSearchCV``.
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=10000,
            solver="lbfgs",
            class_weight="balanced",
        )),
    ])
    param_grid = {"clf__C": [10.0, 15, 18, 21]}
    return pipeline, param_grid


def build_rf_pipeline() -> tuple[Pipeline, dict]:
    """Return a Random Forest pipeline and its hyperparameter grid.

    Returns
    -------
    pipeline : Pipeline
        ``SimpleImputer`` → ``RandomForestClassifier``.
    param_grid : dict
        Grid for ``GridSearchCV``.
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )),
    ])
    param_grid = {
        "clf__n_estimators": [300, 600],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_split": [2, 5, 8, 10],
        "clf__min_samples_leaf": [2, 5],
        "clf__max_features": ["sqrt", 0.5],
    }
    return pipeline, param_grid


def build_xgb_pipeline(
    scale_pos_weight: float = 1.0,
) -> tuple[Pipeline, dict]:
    """Return an XGBoost pipeline and its hyperparameter grid.

    Parameters
    ----------
    scale_pos_weight : float, optional
        Ratio of negative to positive samples.  Pass
        ``(n_neg / n_pos)`` from the training labels so the grid can
        search over ``[1.0, scale_pos_weight]`` (default: 1.0).

    Returns
    -------
    pipeline : Pipeline
        ``SimpleImputer`` → ``XGBClassifier``.
    param_grid : dict
        Grid for ``GridSearchCV``.
    """
    from xgboost import XGBClassifier  # lazy import — optional dependency

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )),
    ])
    param_grid = {
        "clf__n_estimators": [300, 600],
        "clf__learning_rate": [0.03, 0.1],
        "clf__max_depth": [2, 3, 4],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
        "clf__reg_lambda": [1.0, 10.0],
        "clf__min_child_weight": [1, 3, 5, 10],
        "clf__scale_pos_weight": [1.0, scale_pos_weight],
    }
    return pipeline, param_grid


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    pipeline: Pipeline,
    param_grid: dict,
    cv,
    logger: logging.Logger,
    *,
    model_name: str = "model",
) -> tuple[Pipeline, pd.DataFrame, dict]:
    """Tune hyperparameters via ``GridSearchCV`` and refit on all data.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (all seen/training data).
    y : pd.Series
        Binary labels.
    pipeline : Pipeline
        Sklearn pipeline (e.g. from ``build_lr_pipeline``).
    param_grid : dict
        Hyperparameter search space.
    cv : CV splitter
        Any sklearn-compatible cross-validation object (e.g.
        ``MonteCarloCV``).
    logger : logging.Logger
        Logger for progress messages.
    model_name : str, optional
        Human-readable name used in log messages (default: ``"model"``).

    Returns
    -------
    fitted_pipeline : Pipeline
        Pipeline refit on all of *X* / *y* with the best parameters.
    cv_results : pd.DataFrame
        Per-split breakdown with columns ``split``, ``auc``, ``accuracy``,
        ``f1``.
    summary : dict
        Aggregated metrics: mean and std for each of auc, accuracy, f1,
        plus ``best_params``.
    """
    logger.info("Training %s …", model_name)

    scoring = {
        "auc": "roc_auc",
        "accuracy": "accuracy",
        "f1": make_scorer(f1_score),
    }

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit=False,
        n_jobs=-1,
        return_train_score=False,
    )
    grid.fit(X, y)

    # --- Identify best params (ranked by AUC) ----------------------------
    best_idx = grid.cv_results_["rank_test_auc"].argmin()
    best_params = {
        k: grid.cv_results_[f"param_{k}"][best_idx]
        for k in param_grid
    }
    logger.info("%s best params: %s", model_name, best_params)

    # --- Per-split breakdown for the best param combo ---------------------
    n_splits = cv.get_n_splits()
    rows = []
    for i in range(n_splits):
        rows.append({
            "split": i,
            "auc": grid.cv_results_[f"split{i}_test_auc"][best_idx],
            "accuracy": grid.cv_results_[f"split{i}_test_accuracy"][best_idx],
            "f1": grid.cv_results_[f"split{i}_test_f1"][best_idx],
        })
    cv_results = pd.DataFrame(rows)

    summary = {
        "best_params": best_params,
        "auc_mean": cv_results["auc"].mean(),
        "auc_std": cv_results["auc"].std(),
        "accuracy_mean": cv_results["accuracy"].mean(),
        "accuracy_std": cv_results["accuracy"].std(),
        "f1_mean": cv_results["f1"].mean(),
        "f1_std": cv_results["f1"].std(),
    }

    logger.info(
        "%s CV — AUC: %.4f ± %.4f  |  Accuracy: %.4f ± %.4f  |  F1: %.4f ± %.4f",
        model_name,
        summary["auc_mean"], summary["auc_std"],
        summary["accuracy_mean"], summary["accuracy_std"],
        summary["f1_mean"], summary["f1_std"],
    )

    # --- Refit on all data with best params -------------------------------
    fitted = pipeline.set_params(**best_params)
    fitted.fit(X, y)
    logger.info("%s refit on full seen data (%d samples).", model_name, len(X))

    return fitted, cv_results, summary


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_classifier(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    logger: logging.Logger,
    *,
    model_name: str = "model",
) -> dict:
    """Evaluate a fitted pipeline on a held-out test set.

    Parameters
    ----------
    model : Pipeline
        A fitted sklearn ``Pipeline``.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        True labels.
    logger : logging.Logger
        Logger for evaluation output.
    model_name : str, optional
        Human-readable name used in log messages (default: ``"model"``).

    Returns
    -------
    dict
        Keys: ``auc``, ``accuracy``, ``f1``.
    """
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)

    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    logger.info("%s TEST — AUC: %.4f  Accuracy: %.4f  F1: %.4f", model_name, auc, acc, f1)
    logger.info("%s TEST confusion matrix:\n%s", model_name, confusion_matrix(y_test, pred))
    logger.info(
        "%s TEST classification report:\n%s",
        model_name,
        classification_report(y_test, pred, digits=3),
    )

    return {"auc": auc, "accuracy": acc, "f1": f1}
