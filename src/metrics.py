"""Metric helpers for classifier validation."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


LOGGER = logging.getLogger(__name__)


def _has_two_classes(y_true: np.ndarray) -> bool:
	labels = np.unique(np.asarray(y_true))
	return labels.size >= 2


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
	"""Compute ROC AUC, returning NaN when undefined."""
	y_t = np.asarray(y_true)
	y_s = np.asarray(y_score, dtype=float)

	if not _has_two_classes(y_t):
		LOGGER.warning("AUC undefined: y_true has a single class in this fold.")
		return float("nan")

	try:
		return float(roc_auc_score(y_t, y_s))
	except ValueError as exc:
		LOGGER.warning("AUC computation failed: %s", exc)
		return float("nan")


def compute_aupr(y_true: np.ndarray, y_score: np.ndarray) -> float:
	"""Compute AUC-PR, returning NaN when undefined."""
	y_t = np.asarray(y_true)
	y_s = np.asarray(y_score, dtype=float)

	if not _has_two_classes(y_t):
		LOGGER.warning("AUC-PR undefined: y_true has a single class in this fold.")
		return float("nan")

	try:
		return float(average_precision_score(y_t, y_s))
	except ValueError as exc:
		LOGGER.warning("AUC-PR computation failed: %s", exc)
		return float("nan")


__all__ = ["compute_auc", "compute_aupr"]
