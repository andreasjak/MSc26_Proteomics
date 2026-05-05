"""Statistical helpers shared across scripts and notebooks."""

from __future__ import annotations

import math


def nadeau_bengio_se(s: float, n_splits: int, n_train: int, n_test: int) -> float:
    """Nadeau-Bengio corrected SE for repeated random subsampling.

    With B random train/test splits over the same pool of n samples, the per-split
    metrics are positively correlated through shared test patients, so the naive
    SD/sqrt(B) underestimates the SE of the mean generalisation estimate.
    Nadeau & Bengio (2003) propose

        SE_NB = s * sqrt(1/B + n_test / n_train)

    where s is the per-split sample standard deviation of the metric.

    Parameters
    ----------
    s : float
        Per-split sample standard deviation of the metric.
    n_splits : int
        Number of random splits (B).
    n_train : int
        Training-set size per split.
    n_test : int
        Test-set size per split.
    """
    if n_splits <= 0:
        raise ValueError(f"n_splits must be > 0, got {n_splits}.")
    if n_train <= 0:
        raise ValueError(f"n_train must be > 0, got {n_train}.")
    if n_test < 0:
        raise ValueError(f"n_test must be >= 0, got {n_test}.")
    return float(s) * math.sqrt(1.0 / n_splits + n_test / n_train)
