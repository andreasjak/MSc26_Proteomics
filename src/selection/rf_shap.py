"""Random Forest + SHAP based feature selector."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap

from .base import SelectionMethod, validate_selection_output


class RFSHAPSelection(SelectionMethod):
    """Rank proteins by their SHAP-derived importance from a Random Forest classifier.

    Proteins are ranked by their mean positive SHAP value computed exclusively
    over ARDS patients (s_i), which captures proteins that actively drive
    predictions toward ARDS rather than proteins associated with the majority
    non-ARDS class.

    In addition to the ranked list, a native selection is computed as the
    intersection of three ranked lists derived from the SHAP matrix:

        S1 : top-k by mean absolute SHAP over all patients (global importance).
        S2 : top-k by mean SHAP difference between ARDS and non-ARDS patients
             (directional gap between groups).
        S3 : top-k by mean positive SHAP over ARDS patients, i.e. s_i
             (proteins actively pushing predictions toward ARDS).

    The intersection S1 ∩ S2 ∩ S3 forms the ``significant`` mask returned by
    ``select``, where k is controlled by ``native_pool_size``.

    Parameters
    ----------
    native_pool_size:
        Candidate pool size k used when computing the native intersection
        selection. Corresponds to ``--rf-native-pool-size`` in the CLI.
    n_estimators:
        Number of trees in the Random Forest. A large value (default 10 000)
        reduces variance of the SHAP estimates at the cost of compute time.
    """

    name = "rf_shap"

    def __init__(
        self,
        native_pool_size: int = 50,
        n_estimators: int = 10_000,
    ) -> None:
        self._split_seed: int = 42
        self.native_pool_size = native_pool_size
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
            "native_pool_size": self.native_pool_size,
            "n_estimators": self.n_estimators,
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
        """Fit a Random Forest, compute SHAP values, and rank proteins by s_i.

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
            Protein indices sorted by s_i descending (length ``n_proteins``).
        scores:
            s_i for every protein, aligned to the original protein order
            (length ``n_proteins``).
        q_value:
            Placeholder array of ones — no formal significance test is
            performed for this method (length ``n_proteins``).
        significant:
            Boolean mask (length ``n_proteins``) that is ``True`` for proteins
            in the native intersection S1 ∩ S2 ∩ S3.
        """
        n_proteins = X_train.shape[1]
        ards_mask = y_train == 1
        non_ards_mask = ~ards_mask

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
        #    shap_matrix: (n_samples, n_proteins), signed contributions to   #
        #    the ARDS-class prediction.                                       #
        # ------------------------------------------------------------------ #
        explainer = shap.TreeExplainer(rf)
        shap_values_raw = explainer.shap_values(X_train)

        # sklearn RF returns a list [shap_class0, shap_class1]; take class 1.
        if isinstance(shap_values_raw, list):
            shap_matrix = shap_values_raw[1]          # shape (n_samples, n_proteins)
        else:
            # Some shap versions return shape (n_samples, n_proteins, n_classes)
            shap_matrix = shap_values_raw[:, :, 1]

        shap_matrix = np.asarray(shap_matrix, dtype=float)

        # ------------------------------------------------------------------ #
        # 3. Primary score: mean positive SHAP over ARDS patients (s_i)     #
        # ------------------------------------------------------------------ #
        shap_ards = shap_matrix[ards_mask]                          # (n_ards, n_proteins)
        s_i = np.mean(np.maximum(shap_ards, 0.0), axis=0)          # (n_proteins,)

        # ------------------------------------------------------------------ #
        # 4. Native intersection selection                                    #
        # ------------------------------------------------------------------ #
        k = min(self.native_pool_size, n_proteins)

        # S1: top-k by mean absolute SHAP over all patients
        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)       # (n_proteins,)
        s1 = set(np.argsort(mean_abs_shap)[::-1][:k].tolist())

        # S2: top-k by mean SHAP difference (ARDS − non-ARDS)
        mean_shap_ards = np.mean(shap_matrix[ards_mask], axis=0)
        mean_shap_non_ards = np.mean(shap_matrix[non_ards_mask], axis=0)
        delta_i = mean_shap_ards - mean_shap_non_ards               # (n_proteins,)
        s2 = set(np.argsort(delta_i)[::-1][:k].tolist())

        # S3: top-k by s_i (same as primary score)
        s3 = set(np.argsort(s_i)[::-1][:k].tolist())

        # Intersection
        native_indices = s1 & s2 & s3

        # ------------------------------------------------------------------ #
        # 5. Assemble output arrays (all aligned to original protein order)  #
        # ------------------------------------------------------------------ #
        # Ranked indices: descending s_i
        ranked_indices = np.argsort(s_i)[::-1].copy()

        # Scores: s_i per protein in original order
        scores = s_i.copy()

        # q-values: placeholder ones (no formal test available)
        q_value = np.ones(n_proteins, dtype=float)

        # Significant: native intersection mask
        significant = np.zeros(n_proteins, dtype=bool)
        for idx in native_indices:
            significant[idx] = True

        return validate_selection_output(
            ranked_indices,
            scores,
            q_value,
            significant,
            n_proteins,
            self.name,
        )