"""Run per-split protein selection and compute Stage 5 stability outputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    DATA_PROCESSED,
    K_SPLITS,
    RANDOM_SEED,
    RESULTS_DIR,
    STABILITY_MIN_SIZE,
    STABILITY_PI_PRIMARY,
    STABILITY_TOPK_FALLBACK,
    TOPK_VALUES,
)
from src.data_loading import load_cohort_parquet
from src.logging_utils import setup_logging
from src.selection import METHOD_REGISTRY, validate_selection_output
from src.splits import load_splits
from src.stability import compute_stability, stable_set


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

    X, y, protein_ids = load_cohort_parquet(args.cohort_path)
    n_samples, n_proteins = X.shape
    logger.info(
        "Loaded cohort: n_samples=%d n_proteins=%d prevalence=%.4f",
        n_samples,
        n_proteins,
        float(y.mean()),
    )

    splits = load_splits(args.splits_path)
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
        ranked, scores, significant = validate_selection_output(
            ranked_indices=ranked,
            scores=scores,
            significant=significant,
            n_features=n_proteins,
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
    stability_significant_path = out_dir / "stability_significant.parquet"
    stability_topk_path = out_dir / "stability_topk.parquet"
    stable_set_path = out_dir / "stable_set.json"
    meta_path = out_dir / "meta.json"

    selections.to_parquet(selections_path, index=False)

    stability_significant = compute_stability(
        selections=selections,
        n_splits=args.n_splits,
        mode="significant",
    )
    stability_significant.to_parquet(stability_significant_path, index=False)

    topk_frames: list[pd.DataFrame] = []
    for k in TOPK_VALUES:
        k_stability = compute_stability(
            selections=selections,
            n_splits=args.n_splits,
            mode="topk",
            topk=int(k),
        )
        k_stability = k_stability.loc[
            :, ["protein_idx", "protein_id", "frequency", "mean_rank"]
        ]
        k_stability.insert(2, "k", int(k))
        topk_frames.append(k_stability)

    stability_topk = pd.concat(topk_frames, ignore_index=True)
    stability_topk = stability_topk.loc[
        :, ["protein_idx", "protein_id", "k", "frequency", "mean_rank"]
    ]
    stability_topk.to_parquet(stability_topk_path, index=False)

    stable_ids, method_used = stable_set(
        stability_df=stability_significant,
        pi=STABILITY_PI_PRIMARY,
        min_size=STABILITY_MIN_SIZE,
        topk_fallback=STABILITY_TOPK_FALLBACK,
    )

    frequency_lookup = dict(
        zip(
            stability_significant["protein_id"].astype(str),
            stability_significant["frequency"].astype(float),
        )
    )
    stable_payload = {
        "method": method.name,
        "method_used": method_used,
        "pi": float(STABILITY_PI_PRIMARY),
        "min_size": int(STABILITY_MIN_SIZE),
        "topk_fallback": int(STABILITY_TOPK_FALLBACK),
        "n_proteins": int(len(stable_ids)),
        "protein_ids": stable_ids,
        "protein_frequencies": [
            {
                "protein_id": protein_id,
                "frequency": float(frequency_lookup[protein_id]),
            }
            for protein_id in stable_ids
        ],
    }
    with stable_set_path.open("w", encoding="utf-8") as f:
        json.dump(stable_payload, f, indent=2)

    runtime_seconds = float(time.time() - start_time)
    sig_arr = np.asarray(significant_counts, dtype=float)
    sig_freq_arr = stability_significant["frequency"].to_numpy(dtype=float)

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
        "stability_significant_path": str(stability_significant_path),
        "stability_topk_path": str(stability_topk_path),
        "stable_set_path": str(stable_set_path),
        "topk_values": [int(k) for k in TOPK_VALUES],
        "runtime_seconds": runtime_seconds,
        "significant_per_split": {
            "min": int(sig_arr.min()),
            "median": float(np.median(sig_arr)),
            "max": int(sig_arr.max()),
        },
        "significant_frequency": {
            "top": float(sig_freq_arr.max()),
            "n_ge_0_5": int((sig_freq_arr >= 0.5).sum()),
            "n_ge_0_3": int((sig_freq_arr >= 0.3).sum()),
        },
        "stable_set": {
            "method_used": method_used,
            "n_proteins": int(len(stable_ids)),
        },
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved selections: %s", selections_path)
    logger.info("Saved significance stability: %s", stability_significant_path)
    logger.info("Saved top-k stability: %s", stability_topk_path)
    logger.info("Saved stable set: %s", stable_set_path)
    logger.info("Saved metadata: %s", meta_path)
    logger.info(
        "Significant proteins per split: min=%d median=%.1f max=%d",
        int(sig_arr.min()),
        float(np.median(sig_arr)),
        int(sig_arr.max()),
    )
    logger.info(
        "Significance stability: top_frequency=%.3f proteins>=0.3=%d proteins>=0.5=%d",
        float(sig_freq_arr.max()),
        int((sig_freq_arr >= 0.3).sum()),
        int((sig_freq_arr >= 0.5).sum()),
    )
    logger.info(
        "Stable set: method=%s size=%d",
        method_used,
        len(stable_ids),
    )
    logger.info("Total runtime: %.2f s", runtime_seconds)


if __name__ == "__main__":
    main()
