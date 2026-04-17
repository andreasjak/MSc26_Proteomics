"""Synthetic dataset generation for Stage 8 simulation validation."""

from __future__ import annotations

from typing import Any

import numpy as np


SIGNAL_TYPES = {
	"linear",
	"saturation",
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


def _inject_linear(
	X: np.ndarray,
	y: np.ndarray,
	idx: int,
	effect_size: float,
) -> None:
	X[y == 1, idx] += 0.5 * effect_size
	X[y == 0, idx] -= 0.5 * effect_size


def _inject_saturation(
	X: np.ndarray,
	y: np.ndarray,
	idx: int,
	effect_size: float,
	rng: np.random.Generator,
) -> None:
	base = rng.normal(0.0, 1.0, size=y.size)
	shift = np.where(y == 1, effect_size, -effect_size)
	X[:, idx] = np.tanh(base + shift) + 0.05 * rng.normal(size=y.size)


def _inject_u_shape(
	X: np.ndarray,
	y: np.ndarray,
	idx: int,
	effect_size: float,
	rng: np.random.Generator,
) -> None:
	cls1 = y == 1
	cls0 = ~cls1

	a = max(float(effect_size), 0.1)
	sigma = 1.0
	sigma0 = float(np.sqrt(a**2 + sigma**2))

	signs = rng.choice(np.array([-1.0, 1.0]), size=int(cls1.sum()), replace=True)
	X[cls1, idx] = signs * a + rng.normal(0.0, sigma, size=int(cls1.sum()))
	X[cls0, idx] = rng.normal(0.0, sigma0, size=int(cls0.sum()))


def _inject_threshold(
	X: np.ndarray,
	y: np.ndarray,
	idx: int,
	effect_size: float,
	rng: np.random.Generator,
) -> None:
	base = rng.normal(0.0, 1.0, size=y.size)
	active = base > 0.0
	x = base.copy()
	x[active] += effect_size * np.where(y[active] == 1, 1.0, -1.0)
	X[:, idx] = x


def _inject_xor_pair(
	X: np.ndarray,
	y: np.ndarray,
	idx_a: int,
	idx_b: int,
	effect_size: float,
	rng: np.random.Generator,
) -> None:
	n = y.size
	u = rng.normal(0.0, 1.0, size=n)
	v = rng.normal(0.0, 1.0, size=n)

	sign_u = np.sign(u)
	sign_u[sign_u == 0] = 1.0
	abs_u = np.abs(u) + 0.05
	abs_v = np.abs(v) + 0.05

	sign_v = sign_u.copy()
	sign_v[y == 1] *= -1.0

	scale = max(float(effect_size), 0.1)
	X[:, idx_a] = scale * sign_u * abs_u + 0.05 * rng.normal(size=n)
	X[:, idx_b] = scale * sign_v * abs_v + 0.05 * rng.normal(size=n)


def generate_simulated_dataset(
	n: int,
	p: int,
	class_prevalence: float,
	signal_specs: list[dict[str, Any]],
	seed: int,
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

	for spec in signal_specs:
		_validate_signal_spec(spec)

	rng = np.random.default_rng(int(seed))

	y = np.zeros(n, dtype=np.int64)
	n_pos = int(round(n * float(class_prevalence)))
	n_pos = max(1, min(n - 1, n_pos))
	pos_idx = rng.choice(n, size=n_pos, replace=False)
	y[pos_idx] = 1

	X = rng.normal(0.0, 1.0, size=(n, p)).astype(float)

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
				_inject_xor_pair(X, y, a, b, effect_size, rng)
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
				elif signal_type == "saturation":
					_inject_saturation(X, y, j, effect_size, rng)
				elif signal_type == "u_shape":
					_inject_u_shape(X, y, j, effect_size, rng)
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

	return X, y.astype(np.int64), ground_truth


__all__ = ["generate_simulated_dataset", "SIGNAL_TYPES"]
