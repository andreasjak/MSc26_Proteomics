"""Generate and cache stratified train/test splits for Stage 3.

Reads ``cohort.parquet`` from Stage 2 and writes reusable split artifacts:
- data/processed/splits.pkl
- data/processed/splits_meta.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import DATA_PROCESSED, K_SPLITS, K_SPLITS_EXPENSIVE, RANDOM_SEED, TEST_SIZE
from src.logging_utils import setup_logging
from src.splits import generate_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate stratified train/test split cache for downstream stages.",
    )
    parser.add_argument(
        "--cohort-path",
        type=Path,
        default=DATA_PROCESSED / "cohort.parquet",
        help="Path to cohort parquet from Stage 2 (default: data/processed/cohort.parquet).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_PROCESSED,
        help="Output directory for split artifacts (default: data/processed).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=K_SPLITS,
        help=f"Number of stratified splits to generate (default: {K_SPLITS}).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help=f"Fraction of data in each test fold (default: {TEST_SIZE}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Global random seed for split generation (default: {RANDOM_SEED}).",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="If set, write logs to file via shared logging utility.",
    )
    return parser.parse_args()


def main() -> None:
    start = time.time()
    args = parse_args()

    logger = setup_logging(
        save_results=args.save_results,
        log_subdir="generate_splits",
        script_name="01_generate_splits",
    )

    logger.info("Starting 01_generate_splits.py")
    logger.info(
        "Args: cohort_path=%s output_dir=%s n_splits=%d test_size=%.4f seed=%d",
        args.cohort_path,
        args.output_dir,
        args.n_splits,
        args.test_size,
        args.seed,
    )

    if not args.cohort_path.exists():
        raise FileNotFoundError(
            f"Cohort parquet not found at {args.cohort_path}. Run scripts/00_prepare_data.py first."
        )

    cohort = pd.read_parquet(args.cohort_path)
    if "y" not in cohort.columns:
        raise ValueError("Cohort table must contain a 'y' label column.")

    y = cohort["y"].to_numpy(dtype=int)
    unique_labels = np.unique(y)
    if not np.all(np.isin(unique_labels, [0, 1])):
        raise ValueError(
            f"Expected binary labels 0/1 in 'y', found labels: {unique_labels.tolist()}"
        )

    splits = generate_splits(
        y=y,
        k=args.n_splits,
        test_size=args.test_size,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits_path = args.output_dir / "splits.pkl"
    meta_path = args.output_dir / "splits_meta.json"

    with splits_path.open("wb") as f:
        pickle.dump(splits, f, protocol=pickle.HIGHEST_PROTOCOL)

    n_ards_per_test_fold = [int(y[test_idx].sum()) for _, test_idx in splits]
    n_test_per_fold = [int(test_idx.size) for _, test_idx in splits]
    prevalence_per_fold = [
        float(n_ards / n_test) if n_test > 0 else 0.0
        for n_ards, n_test in zip(n_ards_per_test_fold, n_test_per_fold)
    ]

    meta = {
        "seed": int(args.seed),
        "k_splits": int(args.n_splits),
        "k_splits_expensive": int(K_SPLITS_EXPENSIVE),
        "test_size": float(args.test_size),
        "n_samples": int(y.size),
        "n_ards_total": int(y.sum()),
        "n_ards_per_test_fold": n_ards_per_test_fold,
        "n_test_per_fold": n_test_per_fold,
        "test_prevalence_per_fold": prevalence_per_fold,
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    prevalence_arr = np.asarray(prevalence_per_fold, dtype=float)
    ards_count_arr = np.asarray(n_ards_per_test_fold, dtype=float)

    logger.info("Saved split cache: %s", splits_path)
    logger.info("Saved split metadata: %s", meta_path)
    logger.info(
        "Data summary: n_samples=%d, n_ards=%d (%.2f%%)",
        int(y.size),
        int(y.sum()),
        100.0 * (float(y.mean()) if y.size > 0 else 0.0),
    )
    logger.info(
        "Test-fold ARDS counts: min=%d, median=%.1f, max=%d",
        int(ards_count_arr.min()),
        float(np.median(ards_count_arr)),
        int(ards_count_arr.max()),
    )
    logger.info(
        "Test-fold ARDS prevalence: min=%.4f, median=%.4f, max=%.4f",
        float(prevalence_arr.min()),
        float(np.median(prevalence_arr)),
        float(prevalence_arr.max()),
    )
    logger.info(
        "Expensive methods should consume first %d splits from this shared cache.",
        K_SPLITS_EXPENSIVE,
    )
    logger.info("Finished in %.2f s", time.time() - start)


if __name__ == "__main__":
    main()
