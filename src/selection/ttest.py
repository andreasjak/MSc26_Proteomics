"""Welch t-test based feature selector with FDR correction."""

from __future__ import annotations

import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from .base import SelectionMethod


class TTestSelection(SelectionMethod):
	"""Rank proteins by Benjamini-Hochberg adjusted Welch t-test q-values."""

	name = "ttest"

	def __init__(self, q_threshold: float = 0.05) -> None:
		if not 0.0 < q_threshold < 1.0:
			raise ValueError(f"q_threshold must be in (0, 1), got {q_threshold}.")
		self.q_threshold = float(q_threshold)

	def get_params(self) -> dict[str, object]:
		return {
			"q_threshold": self.q_threshold,
		}

	def select(
		self,
		X_train: np.ndarray,
		y_train: np.ndarray,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

		_, p_values = ttest_ind(
			X[pos_mask],
			X[neg_mask],
			axis=0,
			equal_var=False,
			nan_policy="omit",
		)

		p_values = np.asarray(p_values, dtype=float)
		p_values = np.where(np.isfinite(p_values), p_values, 1.0)
		p_values = np.clip(p_values, 0.0, 1.0)

		rejected, q_values, _, _ = multipletests(
			p_values,
			alpha=self.q_threshold,
			method="fdr_bh",
		)

		q_values = np.asarray(q_values, dtype=float)
		rejected = np.asarray(rejected, dtype=bool)

		ranked_indices = np.lexsort((p_values, q_values)).astype(np.int64, copy=False)
		scores = q_values[ranked_indices]
		significant = rejected[ranked_indices]

		return ranked_indices, scores, significant
