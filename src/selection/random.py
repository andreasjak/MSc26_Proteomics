"""Random baseline selector for the Stage 4 framework."""

from __future__ import annotations

import numpy as np

from .base import SelectionMethod


class RandomSelection(SelectionMethod):
	"""Return a random full ranking with a configurable native cutoff."""

	name = "random"

	def __init__(self, n_significant: int = 30) -> None:
		if n_significant < 0:
			raise ValueError(f"n_significant must be >= 0, got {n_significant}.")
		self.n_significant = int(n_significant)
		self._split_seed: int | None = None

	def set_split_seed(self, seed: int) -> None:
		self._split_seed = int(seed)

	def get_params(self) -> dict[str, object]:
		return {
			"n_significant": self.n_significant,
		}

	def select(
		self,
		X_train: np.ndarray,
		y_train: np.ndarray,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		X = np.asarray(X_train)
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

		n_features = X.shape[1]
		rng = np.random.default_rng(self._split_seed)

		ranked_indices = rng.permutation(n_features).astype(np.int64, copy=False)
		# Scores are uniform random and deliberately NOT sorted by rank: the
		# random baseline has no meaningful score, only a random ordering.
		score_pool = rng.uniform(0.0, 1.0, size=n_features)
		scores = score_pool[ranked_indices]

		significant = np.zeros(n_features, dtype=bool)
		m = min(self.n_significant, n_features)
		if m > 0:
			significant[:m] = True

		return ranked_indices, scores, significant
