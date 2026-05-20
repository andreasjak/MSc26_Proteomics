"""Synthetic dataset generation for Stage 8 simulation validation."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


SIGNAL_TYPES = {
	"linear",
	"bounded_variance",
	"u_shape",
	"threshold",
	"xor_pair",
}


def _validate_signal_spec(spec: dict[str, Any]) -> None:
	if "signal_type" not in spec:
		raise ValueError("Each signal spec must include 'signal_type'.")
	if "effect_size" not in spec:
		raise ValueError("Each signal spec must include 'effect_size'.")

	signal_type = str(spec["signal_type"])
	if signal_type not in SIGNAL_TYPES:
		raise ValueError(
			f"Unsupported signal_type '{signal_type}'. Supported: {sorted(SIGNAL_TYPES)}"
		)

	effect = float(spec["effect_size"])
	if effect <= 0:
		raise ValueError(f"effect_size must be > 0, got {effect}.")

	if signal_type == "xor_pair":
		n_pairs = int(spec.get("n_pairs", 1))
		if n_pairs <= 0:
			raise ValueError(f"n_pairs must be > 0 for xor_pair, got {n_pairs}.")
	else:
		n_features = int(spec.get("n_features", 1))
		if n_features <= 0:
			raise ValueError(
				f"n_features must be > 0 for {signal_type}, got {n_features}."
			)


# Injection convention: every injector reads, modifies, and writes back the
# existing column X[:, idx] (or, for xor_pair, X[:, idx_b]). No injector
# overwrites a column with a freshly drawn signal, so any pre-existing
# correlation structure in X is preserved as far as the transform allows.
def _inject_linear(
	X: np.ndarray,
	y: np.ndarray,
	idx: int,
	effect_size: float,
) -> None:
	X[y == 1, idx] += 0.5 * effect_size
	X[y == 0, idx] -= 0.5 * effect_size


def _inject_bounded_variance(
	X: np.ndarray,
	y: np.ndarray,
	idx: int,
	effect_size: float,
) -> None:
	X[y == 1, idx] *= 1.0 + effect_size
	np.clip(X[:, idx], -3.0, 3.0, out=X[:, idx])


def _mixture_cdf(
	x: np.ndarray | float,
	mus: tuple[float, float],
	sigmas: tuple[float, float],
	weights: tuple[float, float],
) -> np.ndarray | float:
	return sum(
		w * norm.cdf(x, loc=m, scale=s)
		for w, m, s in zip(weights, mus, sigmas)
	)


def _mixture_ppf_scalar(
	u: float,
	mus: tuple[float, float],
	sigmas: tuple[float, float],
	weights: tuple[float, float],
	lo: float = -20.0,
	hi: float = 20.0,
	xtol: float = 1e-12,
) -> float:
	return brentq(
		lambda x: _mixture_cdf(x, mus, sigmas, weights) - u,
		lo,
		hi,
		xtol=xtol,
	)


def _normal_to_mixture(
	x: np.ndarray,
	mus: tuple[float, float],
	sigmas: tuple[float, float],
	weights: tuple[float, float],
) -> np.ndarray:
	u = norm.cdf(x)
	u = np.clip(u, 1e-15, 1 - 1e-15)
	return np.array([_mixture_ppf_scalar(ui, mus, sigmas, weights) for ui in u])


def _inject_u_shape(
	X: np.ndarray,
	y: np.ndarray,
	idx: int,
	effect_size: float,
) -> None:
	cls1 = y == 1
	cls0 = ~cls1

	#a = float(effect_size)
	#sigma = 0.5
	#sigma0 = float(np.sqrt(a**2 + sigma**2))
#
	#X[cls0, idx] *= sigma0
	#X[cls1, idx] = _normal_to_mixture(
	#	X[cls1, idx],
	#	mus=(-a, a),
	#	sigmas=(sigma, sigma),
	#	weights=(0.5, 0.5),
	#)

	a = 1.5*float(effect_size)
	sigma = 0.5
	sigma0 = float(np.sqrt(a**2 + sigma**2))

	#X[cls0, idx] *= sigma0
	X[cls1, idx] = _normal_to_mixture(
		X[cls1, idx],
		mus=(-a, a),
		sigmas=(sigma, sigma),
		weights=(0.5, 0.5),
	) / sigma0


def _inject_threshold(
	X: np.ndarray,
	y: np.ndarray,
	idx: int,
	effect_size: float,
	rng: np.random.Generator,
	tau: float = 0.5,
) -> None:
	column = X[:, idx]
	above = np.where(column > tau)[0]
	if above.size < 2:
		return

	score = 2.0 * effect_size * y[above] + rng.standard_normal(above.size)
	sorted_vals = np.sort(column[above])
	rank_of_score = np.argsort(np.argsort(score))
	X[above, idx] = sorted_vals[rank_of_score]


def _inject_xor_pair(
	X: np.ndarray,
	y: np.ndarray,
	idx_a: int,
	idx_b: int,
	effect_size: float,
	rng: np.random.Generator,
	k: int = 3,
) -> None:
	n, d = X.shape
	z = X[:, idx_b]

	others = np.array([j for j in range(d) if j != idx_b])
	zc = z - z.mean()
	cc = X[:, others] - X[:, others].mean(axis=0)
	num = zc @ cc
	den = np.linalg.norm(zc) * np.linalg.norm(cc, axis=0) + 1e-12
	abs_corr = np.abs(num / den)
	k_eff = min(k, others.size)
	partners = others[np.argsort(abs_corr)[-k_eff:]]

	P = X[:, partners]
	beta, *_ = np.linalg.lstsq(P, z, rcond=None)
	fitted = P @ beta
	residual = z - fitted

	sign_a = np.sign(X[:, idx_a])
	sign_a[sign_a == 0] = 1.0
	flip = np.where(y == 1, -1.0, 1.0)
	target_sign = sign_a * flip
	p = 0.5 + 0.5 * float(np.clip(effect_size, 0.0, 1.0))
	follow = rng.uniform(size=n) < p
	final_sign = np.where(follow, target_sign, -target_sign)

	X[:, idx_b] = fitted + final_sign * np.abs(residual)


def _build_block_cov(
	p: int,
	block_size: int,
	rng: np.random.Generator,
	rho_low: float,
	rho_high: float,
) -> np.ndarray:
	"""Block-diagonal correlation matrix with one shared rho per block.

	Kept as a utility for inspection/visualisation. The hot-path sampler
	`_sample_block_correlated` draws each block separately and does not
	build this full matrix.
	"""
	cov = np.eye(p)
	for start in range(0, p, block_size):
		end = start + block_size
		rho = float(rng.uniform(rho_low, rho_high))
		block = np.full((block_size, block_size), rho)
		np.fill_diagonal(block, 1.0)
		cov[start:end, start:end] = block
	return cov


def _sample_block_correlated(
	n: int,
	p: int,
	block_size: int,
	rng: np.random.Generator,
	rho_low: float,
	rho_high: float,
) -> np.ndarray:
	"""Sample (n, p) with block-diagonal exchangeable correlation.

	Each block of size `block_size` uses a shared off-diagonal
	rho ~ U(rho_low, rho_high) sampled once per block, then is drawn via
	`multivariate_normal` on the small block_size x block_size correlation
	matrix. Avoids the O(p^3) Cholesky of the full p x p matrix.
	"""
	n_blocks = p // block_size
	mean_b = np.zeros(block_size)
	block = np.empty((block_size, block_size), dtype=float)
	X = np.empty((n, p), dtype=float)
	for b in range(n_blocks):
		rho = float(rng.uniform(rho_low, rho_high))
		block.fill(rho)
		np.fill_diagonal(block, 1.0)
		start = b * block_size
		end = start + block_size
		X[:, start:end] = rng.multivariate_normal(mean_b, block, size=n)
	return X


def generate_simulated_dataset(
	n: int,
	p: int,
	class_prevalence: float,
	signal_specs: list[dict[str, Any]],
	seed: int,
	block_size: int | None = None,
	rho_low: float = 0.3,
	rho_high: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
	"""Return synthetic X, y and ground-truth signal index mapping.

	Parameters
	----------
	n : int
		Number of samples.
	p : int
		Number of features.
	class_prevalence : float
		Fraction of positive class samples.
	signal_specs : list[dict[str, Any]]
		Signal configuration list. Each entry contains at minimum:
		``signal_type`` and ``effect_size``.
	seed : int
		Random seed for deterministic simulation.

	Returns
	-------
	tuple[np.ndarray, np.ndarray, dict[str, Any]]
		``X`` shape (n, p), ``y`` shape (n,), and ground truth metadata.
	"""
	if n <= 1:
		raise ValueError(f"n must be > 1, got {n}.")
	if p <= 0:
		raise ValueError(f"p must be > 0, got {p}.")
	if not 0.0 < class_prevalence < 1.0:
		raise ValueError(
			f"class_prevalence must be in (0, 1), got {class_prevalence}."
		)
	if not signal_specs:
		raise ValueError("signal_specs must contain at least one signal definition.")

	if block_size is not None:
		if int(block_size) <= 0:
			raise ValueError(f"block_size must be > 0 or None, got {block_size}.")
		if p % int(block_size) != 0:
			raise ValueError(
				f"p={p} must be divisible by block_size={block_size}."
			)
		if not (0.0 <= float(rho_low) <= float(rho_high) < 1.0):
			raise ValueError(
				f"Require 0 <= rho_low <= rho_high < 1, got rho_low={rho_low}, rho_high={rho_high}."
			)

	for spec in signal_specs:
		_validate_signal_spec(spec)

	rng = np.random.default_rng(int(seed))

	y = np.zeros(n, dtype=np.int64)
	n_pos = int(round(n * float(class_prevalence)))
	n_pos = max(1, min(n - 1, n_pos))
	pos_idx = rng.choice(n, size=n_pos, replace=False)
	y[pos_idx] = 1

	if block_size is None:
		X = rng.normal(0.0, 1.0, size=(n, p)).astype(float)
		feature_block: list[int] = [-1] * p
		blocks: list[list[int]] = []
	else:
		bs = int(block_size)
		X = _sample_block_correlated(n, p, bs, rng, float(rho_low), float(rho_high))
		n_blocks = p // bs
		feature_block = [i // bs for i in range(p)]
		blocks = [list(range(b * bs, (b + 1) * bs)) for b in range(n_blocks)]

	available = np.arange(p, dtype=np.int64)
	rng.shuffle(available)
	cursor = 0

	signal_records: list[dict[str, Any]] = []

	def reserve(count: int) -> np.ndarray:
		nonlocal cursor
		if cursor + count > p:
			raise ValueError(
				"Not enough feature slots for requested signal specs. "
				f"Requested at least {cursor + count}, but p={p}."
			)
		out = available[cursor : cursor + count]
		cursor += count
		return out

	for spec in signal_specs:
		signal_type = str(spec["signal_type"])
		effect_size = float(spec["effect_size"])

		if signal_type == "xor_pair":
			n_pairs = int(spec.get("n_pairs", 1))
			idx = reserve(2 * n_pairs)
			for i in range(n_pairs):
				a = int(idx[2 * i])
				b = int(idx[2 * i + 1])
				k = block_size - 1 if block_size is not None else 3
				_inject_xor_pair(X, y, a, b, effect_size, rng, k)
				signal_records.append(
					{
						"signal_type": signal_type,
						"effect_size": effect_size,
						"indices": [a, b],
					}
				)
		else:
			n_features = int(spec.get("n_features", 1))
			idx = reserve(n_features)
			for feature_idx in idx:
				j = int(feature_idx)
				if signal_type == "linear":
					_inject_linear(X, y, j, effect_size)
				elif signal_type == "bounded_variance":
					_inject_bounded_variance(X, y, j, effect_size)
				elif signal_type == "u_shape":
					_inject_u_shape(X, y, j, effect_size)
				elif signal_type == "threshold":
					_inject_threshold(X, y, j, effect_size, rng)
				else:
					raise ValueError(f"Unhandled signal type: {signal_type}")

				signal_records.append(
					{
						"signal_type": signal_type,
						"effect_size": effect_size,
						"indices": [j],
					}
				)

	by_type: dict[str, list[int]] = {k: [] for k in sorted(SIGNAL_TYPES)}
	by_type_effect: dict[str, dict[float, list[int]]] = {k: {} for k in sorted(SIGNAL_TYPES)}

	for record in signal_records:
		signal_type = str(record["signal_type"])
		effect = float(record["effect_size"])
		indices = [int(i) for i in record["indices"]]

		by_type[signal_type].extend(indices)
		by_type_effect[signal_type].setdefault(effect, []).extend(indices)

	by_type = {k: sorted(set(v)) for k, v in by_type.items()}
	for signal_type, effect_map in by_type_effect.items():
		by_type_effect[signal_type] = {
			float(effect): sorted(set(indices))
			for effect, indices in effect_map.items()
		}

	all_signal_indices = sorted({idx for values in by_type.values() for idx in values})
	signal_set = set(all_signal_indices)
	noise_indices = [i for i in range(p) if i not in signal_set]

	ground_truth: dict[str, Any] = {**by_type}
	ground_truth["by_type_effect"] = by_type_effect
	ground_truth["signal_records"] = signal_records
	ground_truth["all_signal_indices"] = all_signal_indices
	ground_truth["noise_indices"] = noise_indices
	ground_truth["block_size"] = (int(block_size) if block_size is not None else None)
	ground_truth["feature_block"] = feature_block
	ground_truth["blocks"] = blocks

	return X, y.astype(np.int64), ground_truth


__all__ = ["generate_simulated_dataset", "SIGNAL_TYPES"]
