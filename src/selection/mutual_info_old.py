"""Mutual-information based feature selector with permutation p-values.

Four variants are supported, all selected via the ``variant`` argument:

* ``"adaptive"``        — Method 1, Besag & Clifford adaptive per-feature stop.
* ``"per_feature_gpd"`` — Method 2, per-feature null + GPD tail extrapolation.
* ``"pooled"``          — Method 3, pooled standardised null, empirical p-values.
* ``"pooled_gpd"``      — Method 4, pooled standardised null + GPD tail.

All variants return BH q-values and the BH rejection mask, matching the
contract of :class:`TTestSelection`.
"""

from __future__ import annotations

import math

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import genpareto
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.multitest import multipletests

from .base import SelectionMethod


_VARIANTS = ("adaptive", "per_feature_gpd", "pooled", "pooled_gpd")


class MutualInfoSelection(SelectionMethod):
	"""Rank proteins by MI with permutation-based BH q-values."""

	name = "mi"

	def __init__(
		self,
		variant: str = "pooled",
		q_threshold: float = 0.05,
		n_permutations: int = 1000,
		hits: int = 5,
		max_permutations: int | None = None,
		tail_quantile: float = 0.90,
		n_neighbors: int = 3,
		n_jobs: int = -1,
		random_state: int | None = None,
	) -> None:
		if variant not in _VARIANTS:
			raise ValueError(f"variant must be one of {_VARIANTS}, got {variant!r}.")
		if not 0.0 < q_threshold < 1.0:
			raise ValueError(f"q_threshold must be in (0, 1), got {q_threshold}.")
		if n_permutations < 1:
			raise ValueError(f"n_permutations must be >= 1, got {n_permutations}.")
		if hits < 1:
			raise ValueError(f"hits must be >= 1, got {hits}.")
		if max_permutations is not None and max_permutations < hits:
			raise ValueError("max_permutations must be >= hits when provided.")
		if not 0.0 < tail_quantile < 1.0:
			raise ValueError(f"tail_quantile must be in (0, 1), got {tail_quantile}.")
		if n_neighbors < 1:
			raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}.")

		self.variant = variant
		self.q_threshold = float(q_threshold)
		self.n_permutations = int(n_permutations)
		self.hits = int(hits)
		self.max_permutations = max_permutations
		self.tail_quantile = float(tail_quantile)
		self.n_neighbors = int(n_neighbors)
		self.n_jobs = int(n_jobs)
		self.random_state = random_state
		self._split_seed: int | None = None
		self.p_values_: np.ndarray | None = None

	def set_split_seed(self, seed: int) -> None:
		self._split_seed = int(seed)

	def get_params(self) -> dict[str, object]:
		return {
			"variant": self.variant,
			"q_threshold": self.q_threshold,
			"n_permutations": self.n_permutations,
			"hits": self.hits,
			"max_permutations": self.max_permutations,
			"tail_quantile": self.tail_quantile,
			"n_neighbors": self.n_neighbors,
			"n_jobs": self.n_jobs,
			"random_state": self.random_state,
		}

	# ------------------------------------------------------------------ select
	def select(
		self,
		X_train: np.ndarray,
		y_train: np.ndarray,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		X = np.asarray(X_train, dtype=float)
		y = np.asarray(y_train).ravel()
		_validate_inputs(X, y)

		base_seed = self._split_seed if self._split_seed is not None else self.random_state

		if self.variant == "adaptive":
			T_obs, p_values = self._select_adaptive(X, y, base_seed)
		else:
			T_obs, M = self._compute_null_matrix(X, y, base_seed)
			if self.variant == "per_feature_gpd":
				p_values = self._pvalues_per_feature_gpd(T_obs, M)
			elif self.variant == "pooled":
				p_values = self._pvalues_pooled(T_obs, M)
			else:  # pooled_gpd
				p_values = self._pvalues_pooled(T_obs, M, gpd=True)

		p_values = np.clip(np.where(np.isfinite(p_values), p_values, 1.0), 0.0, 1.0)
		self.p_values_ = p_values
		return _bh_triple(p_values, T_obs, self.q_threshold)

	# -------------------------------------------------------------- variants
	def _select_adaptive(
		self,
		X: np.ndarray,
		y: np.ndarray,
		base_seed: int | None,
	) -> tuple[np.ndarray, np.ndarray]:
		n_features = X.shape[1]
		if self.max_permutations is None:
			B_max = int(math.ceil(self.hits * n_features / self.q_threshold))
		else:
			B_max = int(self.max_permutations)

		T_obs = _mi_full(X, y, self.n_neighbors, base_seed, n_jobs=self.n_jobs)

		seed_seq = np.random.SeedSequence(base_seed)
		feature_seeds = seed_seq.spawn(n_features)

		results = Parallel(n_jobs=self.n_jobs, backend="loky")(
			delayed(_adaptive_one_feature)(
				X[:, j],
				y,
				T_obs[j],
				self.hits,
				B_max,
				self.n_neighbors,
				feature_seeds[j],
			)
			for j in range(n_features)
		)
		return T_obs, np.asarray(results, dtype=float)

	def _compute_null_matrix(
		self,
		X: np.ndarray,
		y: np.ndarray,
		base_seed: int | None,
	) -> tuple[np.ndarray, np.ndarray]:
		"""Return (T_obs of shape (p,), null matrix M of shape (B, p))."""
		seed_seq = np.random.SeedSequence(base_seed)
		obs_seed, *perm_seeds = seed_seq.spawn(self.n_permutations + 1)

		T_obs = _mi_full(X, y, self.n_neighbors, obs_seed, n_jobs=self.n_jobs)

		M = np.empty((self.n_permutations, X.shape[1]), dtype=float)
		for b, ps in enumerate(perm_seeds):
			rng = np.random.default_rng(ps)
			y_perm = rng.permutation(y)
			M[b] = _mi_full(X, y_perm, self.n_neighbors, ps, n_jobs=self.n_jobs)
		return T_obs, M

	def _pvalues_per_feature_gpd(
		self,
		T_obs: np.ndarray,
		M: np.ndarray,
	) -> np.ndarray:
		B = M.shape[0]
		p_values = Parallel(n_jobs=self.n_jobs, backend="loky")(
			delayed(_pvalue_one_feature_gpd)(
				T_obs[j],
				M[:, j],
				B,
				self.tail_quantile,
			)
			for j in range(M.shape[1])
		)
		return np.asarray(p_values, dtype=float)

	def _pvalues_pooled(
		self,
		T_obs: np.ndarray,
		M: np.ndarray,
		*,
		gpd: bool = False,
	) -> np.ndarray:
		mu = M.mean(axis=0)
		sigma = M.std(axis=0, ddof=1)
		dead = sigma <= 0

		safe_sigma = np.where(dead, 1.0, sigma)
		S = (M - mu) / safe_sigma
		S_obs = (T_obs - mu) / safe_sigma

		pooled_sorted = np.sort(S.ravel())
		N = pooled_sorted.size

		# empirical p: (1 + #{S >= S_obs}) / (N + 1)
		ge_count = N - np.searchsorted(pooled_sorted, S_obs, side="left")
		p_emp = (1.0 + ge_count) / (N + 1.0)

		if gpd:
			u = np.quantile(pooled_sorted, self.tail_quantile)
			exceed_mask = pooled_sorted > u
			exceedances = pooled_sorted[exceed_mask] - u
			N_u = exceedances.size
			fit = _fit_gpd(exceedances) if N_u >= 10 else None
			if fit is not None:
				xi, beta = fit
				above = S_obs > u
				if np.any(above):
					s = S_obs[above] - u
					tail_p = (N_u / N) * np.power(
						np.maximum(1.0 + xi * s / beta, 1e-300),
						-1.0 / xi,
					)
					p_emp[above] = np.clip(tail_p, 0.0, 1.0)

		# Features whose null was constant carry no information.
		p_emp[dead] = 1.0
		return p_emp


# ---------------------------------------------------------------- helpers


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


def _mi_full(
	X: np.ndarray,
	y: np.ndarray,
	n_neighbors: int,
	seed,
	n_jobs: int,
) -> np.ndarray:
	rs = _coerce_random_state(seed)
	return mutual_info_classif(
		X,
		y,
		discrete_features=False,
		n_neighbors=n_neighbors,
		random_state=rs,
		n_jobs=n_jobs,
	)


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
	t_obs: float,
	hits: int,
	B_max: int,
	n_neighbors: int,
	seed_seq: np.random.SeedSequence,
) -> float:
	rng = np.random.default_rng(seed_seq)
	hit_count = 0
	for b in range(1, B_max + 1):
		y_perm = rng.permutation(y)
		t_perm = _mi_single(x, y_perm, n_neighbors, rng.integers(0, 2**31 - 1))
		if t_perm >= t_obs:
			hit_count += 1
			if hit_count == hits:
				return hits / b
	return (hit_count + 1) / (B_max + 1)


def _pvalue_one_feature_gpd(
	t_obs: float,
	null_vals: np.ndarray,
	B: int,
	tail_quantile: float,
) -> float:
	u = float(np.quantile(null_vals, tail_quantile))
	if t_obs <= u:
		hits = int((null_vals >= t_obs).sum())
		return (1.0 + hits) / (B + 1.0)

	exceedances = null_vals[null_vals > u] - u
	N_u = exceedances.size
	if N_u < 10:
		hits = int((null_vals >= t_obs).sum())
		return (1.0 + hits) / (B + 1.0)

	fit = _fit_gpd(exceedances)
	if fit is None:
		hits = int((null_vals >= t_obs).sum())
		return (1.0 + hits) / (B + 1.0)

	xi, beta = fit
	tail = (N_u / B) * float(
		np.power(max(1.0 + xi * (t_obs - u) / beta, 1e-300), -1.0 / xi)
	)
	return float(np.clip(tail, 0.0, 1.0))


def _fit_gpd(exceedances: np.ndarray) -> tuple[float, float] | None:
	try:
		xi, _, beta = genpareto.fit(exceedances, floc=0.0)
	except Exception:
		return None
	if not (np.isfinite(xi) and np.isfinite(beta)) or beta <= 0:
		return None
	return float(xi), float(beta)


def _bh_triple(
	p_values: np.ndarray,
	T_obs: np.ndarray,
	q_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	p_values = np.asarray(p_values, dtype=float)
	p_values = np.where(np.isfinite(p_values), p_values, 1.0)
	p_values = np.clip(p_values, 0.0, 1.0)

	T_obs = np.asarray(T_obs, dtype=float)
	T_obs = np.where(np.isfinite(T_obs), T_obs, 0.0)

	rejected, q_values, _, _ = multipletests(
		p_values,
		alpha=q_threshold,
		method="fdr_bh",
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
