"""Mutual-information based feature selector with adaptive permutation p-values.

Within each feature, the observed and permuted MI estimates share a single
jitter realisation (via a common ``random_state`` for both calls into
``mutual_info_classif``). The KSG tie-breaking noise therefore cancels in
the ``t_perm >= t_obs`` comparison, leaving only the effect of permuting y.

Returns BH q-values and the BH rejection mask matching the contract of
:class:`TTestSelection`.
"""

from __future__ import annotations

import math

import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.multitest import multipletests

from .base import SelectionMethod

_ALLOWED_CORRECTIONS = {"fdr_bh", "fdr_by", "fdr_tsbh", "fdr_tsbky",
 						"bonferroni", "holm", "hommel", "simes-hochberg",
						"sidak", "holm-sidak"}

class MutualInfoSelection(SelectionMethod):
	"""Rank proteins by MI with permutation-based BH q-values."""

	name = "mi"

	def __init__(
		self,
		q_threshold: float = 0.05,
		hits: int = 5,
		max_permutations: int | None = None,
		correction_method: str = "fdr_bh",
		n_neighbors: int = 3,
		n_jobs: int = -1,
		random_state: int | None = 42,
	) -> None:
		if not 0.0 < q_threshold < 1.0:
			raise ValueError(f"q_threshold must be in (0, 1), got {q_threshold}.")
		if hits < 1:
			raise ValueError(f"hits must be >= 1, got {hits}.")
		if max_permutations is not None and max_permutations < hits:
			raise ValueError("max_permutations must be >= hits when provided.")
		if n_neighbors < 1:
			raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}.")
		if correction_method not in _ALLOWED_CORRECTIONS:
			raise ValueError(f"correction_method must be one of {_ALLOWED_CORRECTIONS}")

		self.q_threshold = float(q_threshold)
		self.hits = int(hits)
		self.max_permutations = int(max_permutations) if max_permutations is not None else None
		self.correction_method = correction_method
		self.n_neighbors = int(n_neighbors)
		self.n_jobs = int(n_jobs)
		self.random_state = random_state
		self._split_seed: int | None = None
		self.p_values_: np.ndarray | None = None

	def set_split_seed(self, seed: int) -> None:
		self._split_seed = int(seed)

	def get_params(self) -> dict[str, object]:
		return {
			"q_threshold": self.q_threshold,
			"hits": self.hits,
			"max_permutations": self.max_permutations,
			"correction_method": self.correction_method,
			"n_neighbors": self.n_neighbors,
			"n_jobs": self.n_jobs,
			"random_state": self.random_state,
		}

	def select(
		self,
		X_train: np.ndarray,
		y_train: np.ndarray,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		X = np.asarray(X_train, dtype=float)
		y = np.asarray(y_train).ravel()
		_validate_inputs(X, y)
		
		base_seed = self._split_seed if self._split_seed is not None else self.random_state

		n_features = X.shape[1]
		if self.max_permutations is None:
			B_max = int(math.ceil(self.hits * n_features / self.q_threshold))
		else:
			B_max = int(self.max_permutations)

		seed_seq = np.random.SeedSequence(base_seed)
		feature_seeds = seed_seq.spawn(n_features)

		results = Parallel(n_jobs=self.n_jobs, backend="loky")(
			delayed(_adaptive_one_feature)(
				X[:, j],
				y,
				self.hits,
				B_max,
				self.n_neighbors,
				feature_seeds[j],
			)
			for j in range(n_features)
		)
		T_obs = np.asarray([r[0] for r in results], dtype=float)
		p_values = np.asarray([r[1] for r in results], dtype=float)
		p_values = np.clip(np.where(np.isfinite(p_values), p_values, 1.0), 0.0, 1.0)
		self.p_values_ = p_values

		rejected, q_values, _, _ = multipletests(
			p_values,
			alpha=self.q_threshold,
			method=self.correction_method,
		)
		q_values = np.asarray(q_values, dtype=float)
		rejected = np.asarray(rejected, dtype=bool)

		ranked_indices = np.lexsort(
			(p_values, -T_obs, q_values)
		).astype(np.int64, copy=False)

		scores = T_obs[ranked_indices]
		q_value = q_values[ranked_indices]
		significant = rejected[ranked_indices]

		return ranked_indices, scores, q_value, significant


# -----------------------
# Helpers
# -----------------------

def _validate_inputs(X: np.ndarray, y: np.ndarray) -> None:
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
	if int((y == 1).sum()) == 0 or int((y == 0).sum()) == 0:
		raise ValueError("y_train must contain both classes 0 and 1.")
	
def _mi_single(
	x: np.ndarray,
	y: np.ndarray,
	n_neighbors: int,
	seed,
) -> float:
	rs = _coerce_random_state(seed)
	val = mutual_info_classif(
		x.reshape(-1, 1),
		y,
		discrete_features=False,
		n_neighbors=n_neighbors,
		random_state=rs,
		n_jobs=1,
	)
	return float(val[0])

def _coerce_random_state(seed) -> int | None:
	if seed is None:
		return None
	if isinstance(seed, np.random.SeedSequence):
		return int(seed.generate_state(1)[0])
	return int(seed)

def _adaptive_one_feature(
	x: np.ndarray,
	y: np.ndarray,
	hits: int,
	B_max: int,
	n_neighbors: int,
	seed_seq: np.random.SeedSequence,
) -> tuple[float, float]:
	rng = np.random.default_rng(seed_seq)
	jitter_seed = int(rng.integers(0, 2**31 - 1))
	t_obs = _mi_single(x, y, n_neighbors, jitter_seed)
	hit_count = 0
	for b in range(1, B_max + 1):
		y_perm = rng.permutation(y)
		t_perm = _mi_single(x, y_perm, n_neighbors, jitter_seed)
		if t_perm >= t_obs:
			hit_count += 1
			if hit_count == hits:
				return t_obs, hits / b
	return t_obs, (hit_count + 1) / (B_max + 1)