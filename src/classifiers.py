"""Classifier builders for Stage 6 validation."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import LR_CONFIG, RF_CONFIG, XGB_CONFIG, XGB_SCALE_POS_WEIGHT_MODE

try:
	from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover
	XGBClassifier = None
	_XGB_IMPORT_ERROR = exc
else:
	_XGB_IMPORT_ERROR = None


def _scale_pos_weight(y_train: np.ndarray) -> float:
	y = np.asarray(y_train)
	n_pos = int((y == 1).sum())
	n_neg = int((y == 0).sum())

	if XGB_SCALE_POS_WEIGHT_MODE != "n_neg_over_n_pos":
		raise ValueError(
			"Unsupported XGB_SCALE_POS_WEIGHT_MODE: "
			f"{XGB_SCALE_POS_WEIGHT_MODE}."
		)
	if n_pos == 0:
		raise ValueError("Cannot compute scale_pos_weight: no positive samples in y_train.")
	return float(n_neg / n_pos)


def build_logreg(random_state: int | None = None) -> LogisticRegression:
	"""Build logistic regression classifier from shared config."""
	params = dict(LR_CONFIG)
	if random_state is not None:
		params["random_state"] = int(random_state)
	return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**params)),
    ])


def build_rf(random_state: int | None = None) -> RandomForestClassifier:
	"""Build random forest classifier from shared config."""
	params = dict(RF_CONFIG)
	if random_state is not None:
		params["random_state"] = int(random_state)
	return RandomForestClassifier(**params)


def build_xgb(y_train: np.ndarray, random_state: int | None = None) -> XGBClassifier:
	"""Build XGBoost classifier with fold-specific class-imbalance weighting."""
	if XGBClassifier is None:  # pragma: no cover
		raise ImportError(
			"xgboost is required for build_xgb but could not be imported."
		) from _XGB_IMPORT_ERROR

	params = dict(XGB_CONFIG)
	params["scale_pos_weight"] = _scale_pos_weight(y_train)
	if random_state is not None:
		params["random_state"] = int(random_state)
	return XGBClassifier(**params)


__all__ = ["build_logreg", "build_rf", "build_xgb"]
