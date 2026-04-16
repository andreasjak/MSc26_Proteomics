"""Run per-split protein selection for a chosen method.

Stage 4 outputs full per-split rankings for all proteins.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import DATA_PROCESSED, K_SPLITS, RANDOM_SEED, RESULTS_DIR
from src.logging_utils import setup_logging
from src.selection.base import SelectionMethod
from src.selection.random import RandomSelection
from src.selection.ttest import TTestSelection


MethodFactory = Callable[[argparse.Namespace], SelectionMethod]

METHOD_REGISTRY: dict[str, MethodFactory] = {
    "ttest": lambda _: TTestSelection(),
    "random": lambda args: RandomSelection(n_significant=args.random_significant),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run split-wise protein selection for a registered method.",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=sorted(METHOD_REGISTRY.keys()),
        help="Selection method to run.",
    )
    parser.add_argument(
        "--cohort-path",
        type=Path,
        default=DATA_PROCESSED / "cohort.parquet",
        help="Path to prepared cohort parquet (default: data/processed/cohort.parquet).",
    )
    parser.add_argument(
        "--splits-path",
        type=Path,
        default=DATA_PROCESSED / "splits.pkl",
        help="Path to split cache pickle (default: data/processed/splits.pkl).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "selection",
        help="Base output directory (default: results/selection).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=K_SPLITS,
        help=f"Number of cached splits to process (default: {K_SPLITS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Global seed used for per-split seeds (default: {RANDOM_SEED}).",
    )
    parser.add_argument(
        "--random-significant",
        type=int,
        default=30,
        help="Native-cutoff size for random baseline significant mask.",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="If set, write logs to file via shared logging utility.",
    )
    return parser.parse_args()


def _load_cohort(cohort_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not cohort_path.exists():
        raise FileNotFoundError(
            f"Cohort file not found at {cohort_path}. Run scripts/00_prepare_data.py first."
        )

    cohort = pd.read_parquet(cohort_path)
    if "y" not in cohort.columns:
        raise ValueError("Cohort parquet must contain a 'y' column.")

    protein_cols = [c for c in cohort.columns if c not in {"patient_id", "y"}]
    if not protein_cols:
        raise ValueError("No protein columns found in cohort parquet.")

    y = cohort["y"].to_numpy(dtype=int)
    labels = np.unique(y)
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError(f"Expected binary y in {{0,1}}, found labels {labels.tolist()}.")

    X = cohort[protein_cols].to_numpy(dtype=float)
    protein_ids = np.asarray(protein_cols, dtype=object)
    return X, y, protein_ids


def _load_splits(splits_path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    if not splits_path.exists():
        raise FileNotFoundError(
            f"Split cache not found at {splits_path}. Run scripts/01_generate_splits.py first."
        )

    with splits_path.open("rb") as f:
        splits = pickle.load(f)

    if not isinstance(splits, list) or len(splits) == 0:
        raise ValueError("Split cache must be a non-empty list of (train_idx, test_idx).")

    parsed: list[tuple[np.ndarray, np.ndarray]] = []
    for i, pair in enumerate(splits):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(f"Split #{i} is not a valid (train_idx, test_idx) tuple.")
        train_idx = np.asarray(pair[0], dtype=np.int64)
        test_idx = np.asarray(pair[1], dtype=np.int64)
        parsed.append((train_idx, test_idx))

    return parsed


def _validate_method_output(
    ranked_indices: np.ndarray,
    scores: np.ndarray,
    significant: np.ndarray,
    n_proteins: int,
    method_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ranked = np.asarray(ranked_indices)
    if ranked.shape != (n_proteins,):
        raise ValueError(
            f"{method_name}: ranked_indices must have shape ({n_proteins},), "
            f"got {ranked.shape}."
        )
    if not np.issubdtype(ranked.dtype, np.integer):
        raise ValueError(f"{method_name}: ranked_indices must be integer typed.")
    ranked = ranked.astype(np.int64, copy=False)

    unique = np.unique(ranked)
    if unique.size != n_proteins or int(unique[0]) != 0 or int(unique[-1]) != (n_proteins - 1):
        raise ValueError(
            f"{method_name}: ranked_indices must be a permutation of [0, {n_proteins - 1}]."
        )

    scores_arr = np.asarray(scores, dtype=float)
    if scores_arr.shape != (n_proteins,):
        raise ValueError(
            f"{method_name}: scores must have shape ({n_proteins},), "
            f"got {scores_arr.shape}."
        )

    significant_arr = np.asarray(significant, dtype=bool)
    if significant_arr.shape != (n_proteins,):
        raise ValueError(
            f"{method_name}: significant must have shape ({n_proteins},), "
            f"got {significant_arr.shape}."
        )

    return ranked, scores_arr, significant_arr


def main() -> None:
    start_time = time.time()
    args = parse_args()

    if args.n_splits <= 0:
        raise ValueError(f"--n-splits must be > 0, got {args.n_splits}.")
    if args.random_significant < 0:
        raise ValueError(
            f"--random-significant must be >= 0, got {args.random_significant}."
        )

    logger = setup_logging(
        save_results=args.save_results,
        log_subdir=f"selection/{args.method}",
        script_name="02_run_selection",
    )

    logger.info("Starting 02_run_selection.py")
    logger.info(
        "Args: method=%s cohort_path=%s splits_path=%s output_dir=%s n_splits=%d seed=%d",
        args.method,
        args.cohort_path,
        args.splits_path,
        args.output_dir,
        args.n_splits,
        args.seed,
    )

    X, y, protein_ids = _load_cohort(args.cohort_path)
    n_samples, n_proteins = X.shape
    logger.info(
        "Loaded cohort: n_samples=%d n_proteins=%d prevalence=%.4f",
        n_samples,
        n_proteins,
        float(y.mean()),
    )

    splits = _load_splits(args.splits_path)
    if args.n_splits > len(splits):
        raise ValueError(
            f"Requested n_splits={args.n_splits}, but only {len(splits)} cached splits are available."
        )
    splits = splits[: args.n_splits]

    method = METHOD_REGISTRY[args.method](args)
    logger.info("Using method=%s with params=%s", method.name, method.get_params())

    chunks: list[pd.DataFrame] = []
    significant_counts: list[int] = []
    log_every = max(1, args.n_splits // 5)

    for split_id, (train_idx, _) in enumerate(splits):
        split_seed = int(args.seed + split_id)
        method.set_split_seed(split_seed)

        X_train = X[train_idx]
        y_train = y[train_idx]

        ranked, scores, significant = method.select(X_train=X_train, y_train=y_train)
        ranked, scores, significant = _validate_method_output(
            ranked_indices=ranked,
            scores=scores,
            significant=significant,
            n_proteins=n_proteins,
            method_name=method.name,
        )

        chunk = pd.DataFrame(
            {
                "split_id": np.full(n_proteins, split_id, dtype=np.int64),
                "rank": np.arange(n_proteins, dtype=np.int64),
                "protein_idx": ranked,
                "protein_id": protein_ids[ranked],
                "score": scores,
                "significant": significant,
            }
        )
        chunks.append(chunk)

        n_sig = int(significant.sum())
        significant_counts.append(n_sig)

        if (split_id + 1) % log_every == 0 or (split_id + 1) == args.n_splits:
            logger.info(
                "Processed split %d/%d (split_seed=%d, significant=%d)",
                split_id + 1,
                args.n_splits,
                split_seed,
                n_sig,
            )

    selections = pd.concat(chunks, ignore_index=True)
    expected_rows = n_proteins * args.n_splits
    if int(len(selections)) != expected_rows:
        raise RuntimeError(
            f"Unexpected row count: got {len(selections)}, expected {expected_rows}."
        )

    out_dir = args.output_dir / args.method
    out_dir.mkdir(parents=True, exist_ok=True)
    selections_path = out_dir / "selections.parquet"
    meta_path = out_dir / "meta.json"

    selections.to_parquet(selections_path, index=False)

    runtime_seconds = float(time.time() - start_time)
    sig_arr = np.asarray(significant_counts, dtype=float)

    meta = {
        "method": method.name,
        "method_params": method.get_params(),
        "n_splits": int(args.n_splits),
        "n_samples": int(n_samples),
        "n_proteins": int(n_proteins),
        "n_rows": int(len(selections)),
        "seed": int(args.seed),
        "split_seed_rule": "seed + split_id",
        "cohort_path": str(args.cohort_path),
        "splits_path": str(args.splits_path),
        "output_path": str(selections_path),
        "runtime_seconds": runtime_seconds,
        "significant_per_split": {
            "min": int(sig_arr.min()),
            "median": float(np.median(sig_arr)),
            "max": int(sig_arr.max()),
        },
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved selections: %s", selections_path)
    logger.info("Saved metadata: %s", meta_path)
    logger.info(
        "Significant proteins per split: min=%d median=%.1f max=%d",
        int(sig_arr.min()),
        float(np.median(sig_arr)),
        int(sig_arr.max()),
    )
    logger.info("Total runtime: %.2f s", runtime_seconds)


if __name__ == "__main__":
    main()
