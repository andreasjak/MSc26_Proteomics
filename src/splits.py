"""Split-generation utilities for Stage 3.

Provides deterministic stratified train/test split generation used across the
entire pipeline.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def generate_splits(
	y: np.ndarray,
	k: int,
	test_size: float,
	seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
	"""Return stratified train/test indices for repeated evaluation.

	Parameters
	----------
	y : np.ndarray
		Binary outcome labels with shape ``(n_samples,)``.
	k : int
		Number of splits to generate.
	test_size : float
		Fraction of samples assigned to each test fold.
	seed : int
		Global random seed for deterministic split generation.

	Returns
	-------
	list[tuple[np.ndarray, np.ndarray]]
		List of ``(train_idx, test_idx)`` tuples.
	"""
	y_arr = np.asarray(y)

	if y_arr.ndim != 1:
		raise ValueError(f"Expected y to be 1D, got shape {y_arr.shape}.")
	if y_arr.size == 0:
		raise ValueError("Cannot generate splits for an empty label array.")
	if k <= 0:
		raise ValueError(f"k must be > 0, got {k}.")
	if not 0.0 < test_size < 1.0:
		raise ValueError(f"test_size must be in (0, 1), got {test_size}.")

	splitter = StratifiedShuffleSplit(
		n_splits=k,
		test_size=test_size,
		random_state=seed,
	)

	# The splitter only needs feature matrix shape; labels drive stratification.
	dummy_X = np.zeros((y_arr.shape[0], 1), dtype=np.uint8)

	splits: list[tuple[np.ndarray, np.ndarray]] = []
	for train_idx, test_idx in splitter.split(dummy_X, y_arr):
		splits.append(
			(
				train_idx.astype(np.int64, copy=False),
				test_idx.astype(np.int64, copy=False),
			)
		)

	return splits
