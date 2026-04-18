"""Base interface for protein selection methods."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SelectionMethod(ABC):
	"""Abstract base class for per-split protein selection methods."""

	name: str

	def set_split_seed(self, seed: int) -> None:
		"""Optional hook for methods that need split-level randomness."""
		del seed

	def get_params(self) -> dict[str, object]:
		"""Return method-specific parameters for metadata logging."""
		return {}

	@abstractmethod
	def select(
		self,
		X_train: np.ndarray,
		y_train: np.ndarray,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""Return ranked indices, aligned scores, and aligned significance mask.

		Returns
		-------
		tuple[np.ndarray, np.ndarray, np.ndarray]
			(ranked_indices, scores, significant), each length ``n_proteins``.
		"""
		raise NotImplementedError


def validate_selection_output(
	ranked_indices: np.ndarray,
	scores: np.ndarray,
	significant: np.ndarray,
	n_features: int,
	method_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Validate a method's (ranked, scores, significant) return contract.

	Ensures ``ranked_indices`` is a permutation of ``[0, n_features - 1]`` and
	that ``scores`` and ``significant`` have matching shapes. Returns the
	coerced arrays (int64, float, bool).
	"""
	ranked = np.asarray(ranked_indices)
	if ranked.shape != (n_features,):
		raise ValueError(
			f"{method_name}: ranked_indices must have shape ({n_features},), "
			f"got {ranked.shape}."
		)
	if not np.issubdtype(ranked.dtype, np.integer):
		raise ValueError(f"{method_name}: ranked_indices must be integer typed.")
	ranked = ranked.astype(np.int64, copy=False)

	unique = np.unique(ranked)
	if unique.size != n_features or int(unique[0]) != 0 or int(unique[-1]) != (n_features - 1):
		raise ValueError(
			f"{method_name}: ranked_indices must be a permutation of [0, {n_features - 1}]."
		)

	scores_arr = np.asarray(scores, dtype=float)
	if scores_arr.shape != (n_features,):
		raise ValueError(
			f"{method_name}: scores must have shape ({n_features},), "
			f"got {scores_arr.shape}."
		)

	significant_arr = np.asarray(significant, dtype=bool)
	if significant_arr.shape != (n_features,):
		raise ValueError(
			f"{method_name}: significant must have shape ({n_features},), "
			f"got {significant_arr.shape}."
		)

	return ranked, scores_arr, significant_arr
