"""Run Stage 8 simulation validation for one selection method."""

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
    RANDOM_SEED,
    RESULTS_DIR,
    SIM_BLOCK_SIZE,
    SIM_CLASS_PREVALENCE,
    SIM_EFFECT_SIZES,
    SIM_N,
    SIM_P,
    SIM_REPEATS,
    SIM_RHO_HIGH,
    SIM_RHO_LOW,
    SIM_SIGNALS_PER_TYPE,
    TOPK_VALUES,
)
from src.logging_utils import setup_logging
from src.selection import METHOD_REGISTRY, validate_selection_output
from src.simulation import generate_simulated_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run simulation-based recall/FDP validation for one method.",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=sorted(METHOD_REGISTRY.keys()),
        help="Selection method to evaluate.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=SIM_REPEATS,
        help=f"Number of simulation repeats (default: {SIM_REPEATS}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "simulation",
        help="Base output directory (default: results/simulation).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Global seed. Repeat seed is seed + repeat (default: {RANDOM_SEED}).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=SIM_N,
        help=f"Number of simulated samples per repeat (default: {SIM_N}).",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=SIM_P,
        help=f"Number of simulated features per repeat (default: {SIM_P}).",
    )
    parser.add_argument(
        "--class-prevalence",
        type=float,
        default=SIM_CLASS_PREVALENCE,
        help=f"Positive-class prevalence (default: {SIM_CLASS_PREVALENCE}).",
    )
    parser.add_argument(
        "--signals-per-type",
        type=int,
        default=SIM_SIGNALS_PER_TYPE,
        help=f"Signals per type per effect size (default: {SIM_SIGNALS_PER_TYPE}).",
    )
    parser.add_argument(
        "--effect-sizes",
        type=float,
        nargs="*",
        default=None,
        help="Optional list of effect sizes. Defaults to SIM_EFFECT_SIZES from config.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=(SIM_BLOCK_SIZE if SIM_BLOCK_SIZE is not None else 0),
        help=(
            "Block size for block-diagonal noise covariance. "
            "Pass 0 to disable (independent features). "
            f"Default: {SIM_BLOCK_SIZE}."
        ),
    )
    parser.add_argument(
        "--rho-low",
        type=float,
        default=SIM_RHO_LOW,
        help=f"Lower bound for per-block rho ~ U(low, high). Default: {SIM_RHO_LOW}.",
    )
    parser.add_argument(
        "--rho-high",
        type=float,
        default=SIM_RHO_HIGH,
        help=f"Upper bound for per-block rho ~ U(low, high). Default: {SIM_RHO_HIGH}.",
    )
    parser.add_argument(
        "--random-significant",
        type=int,
        default=20,
        help="Native-cutoff size for random baseline significant mask.",
    )
    parser.add_argument(
        "--rf-native-pool-size",
        type=int,
        default=50,
        help="Pool size for random forest native selection.",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="If set, write logs to file via shared logging utility.",
    )
    return parser.parse_args()


def _default_signal_specs(
    effect_sizes: list[float],
    signals_per_type: int,
) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for effect in effect_sizes:
        for signal_type in ["linear", "bounded_variance", "u_shape", "threshold"]:
            specs.append(
                {
                    "signal_type": signal_type,
                    "effect_size": float(effect),
                    "n_features": int(signals_per_type),
                }
            )
        specs.append(
            {
                "signal_type": "xor_pair",
                "effect_size": float(effect),
                "n_pairs": int(signals_per_type),
            }
        )
    return specs


def _aggregate_truth_by_type_effect(
    ground_truth: dict[str, object],
) -> dict[tuple[str, float], set[int]]:
    output: dict[tuple[str, float], set[int]] = {}
    by_type_effect = ground_truth.get("by_type_effect", {})
    if not isinstance(by_type_effect, dict):
        raise ValueError("ground_truth['by_type_effect'] is missing or malformed.")

    for signal_type, effect_map in by_type_effect.items():
        if not isinstance(effect_map, dict):
            continue
        for effect, indices in effect_map.items():
            key = (str(signal_type), float(effect))
            output[key] = set(int(i) for i in indices)
    return output


def _fdp(selected: np.ndarray, noise_set: set[int]) -> float:
    if selected.size == 0:
        return float("nan")
    n_noise = int(sum(int(i) in noise_set for i in selected.tolist()))
    return float(n_noise / selected.size)


def _block_recall(
    selected: np.ndarray,
    truth_indices: set[int],
    feature_block: list[int],
    blocks: list[list[int]],
) -> float:
    if not truth_indices:
        return float("nan")
    selected_set = set(int(i) for i in selected.tolist())
    if not feature_block or not blocks:
        n_found = int(sum(1 for i in truth_indices if i in selected_set))
        return float(n_found / len(truth_indices))
    hits = 0
    for i in truth_indices:
        b = feature_block[int(i)]
        if b < 0:
            hits += int(int(i) in selected_set)
        elif selected_set.intersection(blocks[b]):
            hits += 1
    return float(hits / len(truth_indices))


def _block_fdp(
    selected: np.ndarray,
    signal_blocks: set[int],
    feature_block: list[int],
    noise_set: set[int],
) -> float:
    if selected.size == 0:
        return float("nan")
    if not feature_block or not signal_blocks:
        return _fdp(selected, noise_set)
    n_false = int(
        sum(
            1
            for i in selected.tolist()
            if feature_block[int(i)] not in signal_blocks
        )
    )
    return float(n_false / selected.size)


def main() -> None:
    start_time = time.time()
    args = parse_args()

    if args.n_repeats <= 0:
        raise ValueError(f"--n-repeats must be > 0, got {args.n_repeats}.")
    if args.n_samples <= 1:
        raise ValueError(f"--n-samples must be > 1, got {args.n_samples}.")
    if args.n_features <= 0:
        raise ValueError(f"--n-features must be > 0, got {args.n_features}.")
    if not 0.0 < args.class_prevalence < 1.0:
        raise ValueError(
            f"--class-prevalence must be in (0, 1), got {args.class_prevalence}."
        )
    if args.signals_per_type <= 0:
        raise ValueError(
            f"--signals-per-type must be > 0, got {args.signals_per_type}."
        )
    if args.block_size < 0:
        raise ValueError(f"--block-size must be >= 0, got {args.block_size}.")
    block_size_arg: int | None = args.block_size if args.block_size > 0 else None
    if block_size_arg is not None:
        if args.n_features % block_size_arg != 0:
            raise ValueError(
                f"--n-features ({args.n_features}) must be divisible by "
                f"--block-size ({block_size_arg})."
            )
        if not (0.0 <= args.rho_low <= args.rho_high < 1.0):
            raise ValueError(
                f"Require 0 <= --rho-low <= --rho-high < 1, "
                f"got rho_low={args.rho_low}, rho_high={args.rho_high}."
            )

    effect_sizes = (
        [float(x) for x in args.effect_sizes]
        if args.effect_sizes is not None and len(args.effect_sizes) > 0
        else [float(x) for x in SIM_EFFECT_SIZES]
    )
    if any(x <= 0 for x in effect_sizes):
        raise ValueError(f"All effect sizes must be > 0, got {effect_sizes}.")

    logger = setup_logging(
        save_results=args.save_results,
        log_subdir=f"simulation/{args.method}",
        script_name="05_run_simulation",
    )

    logger.info("Starting 05_run_simulation.py")
    logger.info(
        "Args: method=%s n_repeats=%d n_samples=%d n_features=%d class_prevalence=%.3f signals_per_type=%d block_size=%s rho_low=%.3f rho_high=%.3f effect_sizes=%s seed=%d",
        args.method,
        args.n_repeats,
        args.n_samples,
        args.n_features,
        args.class_prevalence,
        args.signals_per_type,
        block_size_arg,
        args.rho_low,
        args.rho_high,
        effect_sizes,
        args.seed,
    )

    method = METHOD_REGISTRY[args.method](args)
    logger.info("Using method=%s with params=%s", method.name, method.get_params())

    signal_specs = _default_signal_specs(
        effect_sizes=effect_sizes,
        signals_per_type=int(args.signals_per_type),
    )

    recall_rows: list[dict[str, object]] = []
    fdp_rows: list[dict[str, object]] = []
    log_every = max(1, args.n_repeats // 5)

    for repeat in range(1, args.n_repeats + 1):
        repeat_seed = int(args.seed + repeat)
        method.set_split_seed(repeat_seed)

        X, y, truth = generate_simulated_dataset(
            n=int(args.n_samples),
            p=int(args.n_features),
            class_prevalence=float(args.class_prevalence),
            signal_specs=signal_specs,
            seed=repeat_seed,
            block_size=block_size_arg,
            rho_low=float(args.rho_low),
            rho_high=float(args.rho_high),
        )

        ranked, scores, q_value, significant = method.select(X_train=X, y_train=y)
        ranked, _, _, significant = validate_selection_output(
            ranked_indices=ranked,
            scores=scores,
            q_value=q_value,
            significant=significant,
            n_features=int(args.n_features),
            method_name=method.name,
        )

        truth_map = _aggregate_truth_by_type_effect(truth)
        noise_set = set(int(i) for i in truth.get("noise_indices", []))
        feature_block = [int(b) for b in truth.get("feature_block", [])]
        blocks = [[int(i) for i in blk] for blk in truth.get("blocks", [])]
        signal_blocks: set[int] = set()
        if feature_block:
            for i in truth.get("all_signal_indices", []):
                b = feature_block[int(i)]
                if b >= 0:
                    signal_blocks.add(int(b))

        selections: list[tuple[str, np.ndarray]] = []
        for k in TOPK_VALUES:
            k_int = int(k)
            top_idx = ranked[: min(k_int, ranked.size)]
            selections.append((str(k_int), top_idx))

        native_idx = ranked[significant]
        selections.append(("native", native_idx))

        for k_label, selected_idx in selections:
            fdp_rows.append(
                {
                    "repeat": int(repeat),
                    "k": str(k_label),
                    "fdp": float(_fdp(selected_idx, noise_set)),
                    "block_fdp": float(
                        _block_fdp(selected_idx, signal_blocks, feature_block, noise_set)
                    ),
                    "n_selected": int(selected_idx.size),
                }
            )

            for (signal_type, effect_size), truth_indices in truth_map.items():
                denom = len(truth_indices)
                if denom == 0:
                    recall = float("nan")
                else:
                    n_found = int(
                        sum(int(i) in truth_indices for i in selected_idx.tolist())
                    )
                    recall = float(n_found / denom)

                recall_rows.append(
                    {
                        "repeat": int(repeat),
                        "signal_type": str(signal_type),
                        "effect_size": float(effect_size),
                        "k": str(k_label),
                        "recall": float(recall),
                        "block_recall": float(
                            _block_recall(selected_idx, truth_indices, feature_block, blocks)
                        ),
                    }
                )

        if repeat % log_every == 0 or repeat == args.n_repeats:
            logger.info(
                "Processed repeat %d/%d (seed=%d, native_selected=%d)",
                repeat,
                args.n_repeats,
                repeat_seed,
                int(native_idx.size),
            )

    recall_df = (
        pd.DataFrame(recall_rows)
        .loc[:, ["repeat", "signal_type", "effect_size", "k", "recall", "block_recall"]]
        .sort_values(
            ["repeat", "signal_type", "effect_size", "k"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    fdp_df = (
        pd.DataFrame(fdp_rows)
        .loc[:, ["repeat", "k", "fdp", "block_fdp", "n_selected"]]
        .sort_values(["repeat", "k"], kind="mergesort")
        .reset_index(drop=True)
    )

    out_dir = args.output_dir / args.method
    out_dir.mkdir(parents=True, exist_ok=True)
    recall_path = out_dir / "recall.parquet"
    fdp_path = out_dir / "fdp.parquet"
    meta_path = out_dir / "meta.json"

    recall_df.to_parquet(recall_path, index=False)
    fdp_df.to_parquet(fdp_path, index=False)

    runtime_seconds = float(time.time() - start_time)
    meta = {
        "method": args.method,
        "method_params": method.get_params(),
        "n_repeats": int(args.n_repeats),
        "n_samples": int(args.n_samples),
        "n_features": int(args.n_features),
        "class_prevalence": float(args.class_prevalence),
        "signals_per_type": int(args.signals_per_type),
        "block_size": (int(block_size_arg) if block_size_arg is not None else None),
        "rho_low": float(args.rho_low),
        "rho_high": float(args.rho_high),
        "effect_sizes": [float(x) for x in effect_sizes],
        "topk_values": [int(k) for k in TOPK_VALUES],
        "includes_native": True,
        "seed": int(args.seed),
        "repeat_seed_rule": "seed + repeat",
        "recall_path": str(recall_path),
        "fdp_path": str(fdp_path),
        "n_recall_rows": int(len(recall_df)),
        "n_fdp_rows": int(len(fdp_df)),
        "runtime_seconds": runtime_seconds,
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved recall results: %s (rows=%d)", recall_path, len(recall_df))
    logger.info("Saved FDP results: %s (rows=%d)", fdp_path, len(fdp_df))
    logger.info("Saved metadata: %s", meta_path)
    logger.info("Finished in %.2f s", runtime_seconds)


if __name__ == "__main__":
    main()
