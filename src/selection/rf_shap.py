"""Random Forest + SHAP based feature selector."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap

from .base import SelectionMethod


class RFSHAPSelection(SelectionMethod):
    """Rank proteins by mean absolute SHAP value from a Random Forest classifier.

    Proteins are ranked by their mean absolute SHAP value computed over all
    patients (s_j), which reflects each protein's overall contribution to the
    model's predictions regardless of direction or class.

    No native selection is computed per split. Instead, a stable native set
    is derived downstream via stability selection across all resampling splits
    (see the full-data pipeline), where proteins are selected if they appear
    in the top-k_0 ranked proteins in more than half of all splits.

    Parameters
    ----------
    n_estimators:
        Number of trees in the Random Forest. A large value (default 10 000)
        reduces variance of the SHAP estimates at the cost of compute time.
        An empirical stability analysis showed that with 500 trees the overlap
        in top-25 ranked proteins between independent runs was approximately
        44%, whereas with 10 000 trees this increased to approximately 80%.
    """

    name = "rf_shap"

    def __init__(
        self,
        n_estimators: int = 10_000,
    ) -> None:
        self._split_seed: int = 42
        self.n_estimators = n_estimators

    # ------------------------------------------------------------------
    # Base-class hooks
    # ------------------------------------------------------------------

    def set_split_seed(self, seed: int) -> None:
        """Store the per-split seed used to initialise the Random Forest."""
        self._split_seed = seed

    def get_params(self) -> dict[str, object]:
        """Return method-specific parameters for metadata logging."""
        return {
            "n_estimators": self.n_estimators,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "feature_perturbation": "tree_path_dependent",
            "random_state": self._split_seed,
        }

    # ------------------------------------------------------------------
    # Core selection logic
    # ------------------------------------------------------------------

    def select(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fit a Random Forest, compute SHAP values, and rank proteins by s_j.

        Parameters
        ----------
        X_train:
            Feature matrix of shape ``(n_samples, n_proteins)``.
        y_train:
            Binary label vector of shape ``(n_samples,)``, where 1 indicates
            an ARDS patient.

        Returns
        -------
        ranked_indices:
            Protein indices sorted by s_j descending (length ``n_proteins``).
        scores:
            s_j for every protein, in ranked order (length ``n_proteins``).
        q_value:
            All NaN — no formal significance test is performed for this
            method (length ``n_proteins``).
        significant:
            All False — native selection is handled downstream via stability
            selection across splits (length ``n_proteins``).
        """
        n_proteins = X_train.shape[1]

        # ------------------------------------------------------------------ #
        # 1. Fit Random Forest                                                #
        # ------------------------------------------------------------------ #
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_features="sqrt",
            class_weight="balanced",
            random_state=self._split_seed,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        # ------------------------------------------------------------------ #
        # 2. Compute SHAP values via TreeSHAP                                #
        #    tree_path_dependent perturbation is used because proteins are   #
        #    highly correlated in high-dimensional proteomics data, making   #
        #    the marginal distribution a poor approximation of the true      #
        #    conditional distribution.                                        #
        #    shap_matrix: (n_samples, n_proteins), signed contributions to   #
        #    the ARDS-class prediction.                                       #
        # ------------------------------------------------------------------ #
        explainer = shap.TreeExplainer(
            rf,
            feature_perturbation="tree_path_dependent",
        )
        shap_values_raw = explainer.shap_values(
            X_train,
            check_additivity=False,
        )

        # sklearn RF returns a list [shap_class0, shap_class1]; take class 1.
        if isinstance(shap_values_raw, list):
            shap_matrix = shap_values_raw[1]          # (n_samples, n_proteins)
        elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
            shap_matrix = shap_values_raw[:, :, 1]    # (n_samples, n_proteins)
        else:
            shap_matrix = shap_values_raw

        shap_matrix = np.asarray(shap_matrix, dtype=float)

        # ------------------------------------------------------------------ #
        # 3. Primary score: mean absolute SHAP over all patients             #
        #                                                                     #
        #    s_j = (1/n) * sum_i |phi_ij|                                   #
        #                                                                     #
        #    Reflects each protein's overall contribution to the model's     #
        #    predictions regardless of direction or class.                    #
        # ------------------------------------------------------------------ #
        s_j = np.abs(shap_matrix).mean(axis=0)                      # (n_proteins,)

        # ------------------------------------------------------------------ #
        # 4. Assemble output arrays                                           #
        # ------------------------------------------------------------------ #
        ranked_indices = np.argsort(s_j)[::-1].copy()
        scores         = s_j[ranked_indices]
        q_value        = np.full(n_proteins, np.nan, dtype=float)
        significant    = np.zeros(n_proteins, dtype=bool)

        return ranked_indices, scores, q_value, significant 