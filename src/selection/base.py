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
