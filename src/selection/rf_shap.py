"""Random Forest + SHAP based feature selector."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap

from .base import SelectionMethod


class RFSHAPSelection(SelectionMethod):
    """Rank proteins by mean absolute SHAP value from a Random Forest.

    Uses fixed, literature-motivated hyperparameters.
    No significance testing — relies on top-k stability only.
    split_seed varies per split via set_split_seed(), which means
    each of the 20 outer splits gets a different RF initialisation —
    this is the primary robustness mechanism instead of grid search.
    """

    name = "rf_shap"

    def __init__(self) -> None:
        self._split_seed = 42

    def get_params(self) -> dict[str, object]:
        return {
            "n_estimators": 500,
            "max_depth": None,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "random_state": self._split_seed,
        }

    def set_split_seed(self, seed: int) -> None:
        self._split_seed = int(seed)

    def select(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train)

        if X.ndim != 2:
            raise ValueError(f"Expected X_train to be 2D, got shape {X.shape}.")
        if y.ndim != 1:
            raise ValueError(f"Expected y_train to be 1D, got shape {y.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X_train and y_train must have matching row counts, "
                f"got {X.shape[0]} and {y.shape[0]}."
            )
        if X.shape[1] == 0:
            raise ValueError("X_train must contain at least one protein feature.")

        pos_mask = y == 1
        neg_mask = y == 0
        if int(pos_mask.sum()) == 0 or int(neg_mask.sum()) == 0:
            raise ValueError("y_train must contain both classes 0 and 1.")

        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            max_features="sqrt",
            class_weight="balanced",
            random_state=self._split_seed,
            n_jobs=-1,
        )
        rf.fit(X, y)

        explainer = shap.TreeExplainer(
            rf,
            feature_perturbation="tree_path_dependent",
        )
        shap_values = explainer.shap_values(X, check_additivity=False)

        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # newer SHAP: shape is (n_samples, n_features, n_classes)
            shap_class1 = np.abs(shap_values[:, :, 1])
        elif isinstance(shap_values, list):
            # older SHAP: list of [class_0_array, class_1_array]
            shap_class1 = np.abs(shap_values[1])
        else:
            shap_class1 = np.abs(shap_values)

        mean_abs_shap = shap_class1.mean(axis=0)   # shape: (n_proteins,)

        ranked_indices = np.argsort(-mean_abs_shap).astype(np.int64)
        scores         = mean_abs_shap[ranked_indices]

        # No significance — q_value NaN, significant all False
        q_value    = np.full(len(ranked_indices), np.nan, dtype=float)
        significant = np.zeros(len(ranked_indices), dtype=bool)

        return ranked_indices, scores, q_value, significant