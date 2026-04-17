"""Run Stage 9 cross-method aggregation over Stage 5-8 outputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import RESULTS_DIR
from src.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Stage 5-8 outputs into Stage 9 comparison summaries.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="*",
        default=["ttest", "random"],
        help="Methods to aggregate (default: ttest random).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Base results directory (default: results).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "comparison",
        help="Output directory for Stage 9 summaries (default: results/comparison).",
    )
    parser.add_argument(
        "--pi-threshold",
        type=float,
        default=0.3,
        help="Stability frequency threshold for stable-size summary (default: 0.3).",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="If set, fail when any expected input file is missing.",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="If set, write logs to file via shared logging utility.",
    )
    return parser.parse_args()


def _check_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def _load_parquet(
    path: Path,
    label: str,
    strict_missing: bool,
    logger,
) -> pd.DataFrame | None:
    if not path.exists():
        message = f"Missing {label}: {path}"
        if strict_missing:
            raise FileNotFoundError(message)
        logger.warning(message)
        return None
    return pd.read_parquet(path)


def _load_json(path: Path, strict_missing: bool, logger) -> dict | None:
    if not path.exists():
        message = f"Missing JSON: {path}"
        if strict_missing:
            raise FileNotFoundError(message)
        logger.warning(message)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_classifier_summary(
    methods: list[str],
    results_dir: Path,
    output_dir: Path,
    strict_missing: bool,
    logger,
) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    missing_methods: list[str] = []

    required = {"split_id", "classifier", "k", "auc", "aupr"}

    for method in methods:
        scores_path = results_dir / "classifier" / method / "scores.parquet"
        scores = _load_parquet(
            path=scores_path,
            label=f"classifier scores for method='{method}'",
            strict_missing=strict_missing,
            logger=logger,
        )
        if scores is None:
            missing_methods.append(method)
            continue

        _check_columns(scores, required=required, label=str(scores_path))
        chunk = scores.loc[:, ["split_id", "classifier", "k", "auc", "aupr"]].copy()
        chunk["method"] = method
        chunk["k"] = chunk["k"].astype(str)
        frames.append(chunk)

    if not frames:
        raise ValueError("No classifier inputs available. Cannot create classifier_summary.parquet.")

    all_scores = pd.concat(frames, ignore_index=True)
    summary = (
        all_scores.groupby(["method", "classifier", "k"], observed=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_sd=("auc", "std"),
            aupr_mean=("aupr", "mean"),
            aupr_sd=("aupr", "std"),
            n_splits=("split_id", "nunique"),
        )
        .reset_index()
        .sort_values(["method", "classifier", "k"], kind="mergesort")
        .reset_index(drop=True)
    )

    out_path = output_dir / "classifier_summary.parquet"
    summary.to_parquet(out_path, index=False)
    logger.info("Saved classifier summary: %s (rows=%d)", out_path, len(summary))
    return summary, missing_methods


def aggregate_stability_summary(
    methods: list[str],
    results_dir: Path,
    output_dir: Path,
    pi_threshold: float,
    strict_missing: bool,
    logger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[str]], list[str]]:
    rows: list[dict[str, object]] = []
    curve_frames: list[pd.DataFrame] = []
    stable_sets: dict[str, set[str]] = {}
    missing_methods: list[str] = []

    required = {"protein_id", "frequency"}

    for method in methods:
        stability_path = results_dir / "selection" / method / "stability_significant.parquet"
        stable_set_path = results_dir / "selection" / method / "stable_set.json"

        stability = _load_parquet(
            path=stability_path,
            label=f"stability_significant for method='{method}'",
            strict_missing=strict_missing,
            logger=logger,
        )
        if stability is None:
            missing_methods.append(method)
            continue

        _check_columns(stability, required=required, label=str(stability_path))
        stability = stability.loc[:, ["protein_id", "frequency"]].copy()
        stability["protein_id"] = stability["protein_id"].astype(str)
        stability["frequency"] = pd.to_numeric(stability["frequency"], errors="coerce")

        curve = stability.sort_values(
            ["frequency", "protein_id"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        curve = curve.loc[:, ["frequency"]]
        curve.insert(0, "rank", np.arange(1, len(curve) + 1, dtype=np.int64))
        curve.insert(0, "method", method)
        curve_frames.append(curve)

        top_frequency = float(stability["frequency"].max()) if len(stability) > 0 else float("nan")
        n_ge_05 = int((stability["frequency"] >= 0.5).sum())
        n_ge_03 = int((stability["frequency"] >= 0.3).sum())
        stable_size_pi = int((stability["frequency"] >= pi_threshold).sum())

        stable_set_payload = _load_json(
            path=stable_set_path,
            strict_missing=strict_missing,
            logger=logger,
        )

        stable_ids: list[str] = []
        stable_size_saved = None
        stable_method_used = None
        if stable_set_payload is not None:
            stable_ids = [str(x) for x in stable_set_payload.get("protein_ids", [])]
            stable_size_saved = int(stable_set_payload.get("n_proteins", len(stable_ids)))
            stable_method_used = stable_set_payload.get("method_used")

        stable_sets[method] = set(stable_ids)

        rows.append(
            {
                "method": method,
                "stable_size_pi_0_3": stable_size_pi,
                "top_frequency": top_frequency,
                "n_ge_0_5": n_ge_05,
                "n_ge_0_3": n_ge_03,
                "stable_set_size_saved": stable_size_saved,
                "stable_set_method_used": stable_method_used,
            }
        )

    if not rows:
        raise ValueError("No stability inputs available. Cannot create stability_summary.parquet.")

    summary = pd.DataFrame(rows).sort_values(["method"], kind="mergesort").reset_index(drop=True)

    out_path = output_dir / "stability_summary.parquet"
    summary.to_parquet(out_path, index=False)
    logger.info("Saved stability summary: %s (rows=%d)", out_path, len(summary))

    if not curve_frames:
        raise ValueError(
            "No stability-curve inputs available. Cannot create selection_frequency_curve.parquet."
        )

    frequency_curve = pd.concat(curve_frames, ignore_index=True)
    curve_path = output_dir / "selection_frequency_curve.parquet"
    frequency_curve.to_parquet(curve_path, index=False)
    logger.info("Saved selection frequency curve: %s (rows=%d)", curve_path, len(frequency_curve))

    return summary, frequency_curve, stable_sets, missing_methods


def aggregate_enrichment_summary(
    methods: list[str],
    results_dir: Path,
    output_dir: Path,
    strict_missing: bool,
    logger,
) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    missing_methods: list[str] = []
    required = {"library", "bes_raw", "bes_z", "bes_p_emp", "n_significant_terms"}

    for method in methods:
        summary_path = results_dir / "enrichment" / method / "summary.parquet"
        enrichment = _load_parquet(
            path=summary_path,
            label=f"enrichment summary for method='{method}'",
            strict_missing=strict_missing,
            logger=logger,
        )
        if enrichment is None:
            missing_methods.append(method)
            continue

        _check_columns(enrichment, required=required, label=str(summary_path))
        chunk = enrichment.loc[
            :, ["library", "bes_raw", "bes_z", "bes_p_emp", "n_significant_terms"]
        ].copy()
        chunk["method"] = method
        chunk = chunk.rename(columns={"bes_raw": "bes"})
        frames.append(chunk)

    if not frames:
        raise ValueError("No enrichment inputs available. Cannot create enrichment_summary.parquet.")

    summary = (
        pd.concat(frames, ignore_index=True)
        .loc[:, ["method", "library", "bes", "bes_z", "bes_p_emp", "n_significant_terms"]]
        .sort_values(["method", "library"], kind="mergesort")
        .reset_index(drop=True)
    )

    out_path = output_dir / "enrichment_summary.parquet"
    summary.to_parquet(out_path, index=False)
    logger.info("Saved enrichment summary: %s (rows=%d)", out_path, len(summary))
    return summary, missing_methods


def aggregate_simulation_summary(
    methods: list[str],
    results_dir: Path,
    output_dir: Path,
    strict_missing: bool,
    logger,
) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    missing_methods: list[str] = []
    required = {"repeat", "signal_type", "effect_size", "k", "recall", "fdr"}

    for method in methods:
        results_path = results_dir / "simulation" / method / "results.parquet"
        simulation = _load_parquet(
            path=results_path,
            label=f"simulation results for method='{method}'",
            strict_missing=strict_missing,
            logger=logger,
        )
        if simulation is None:
            missing_methods.append(method)
            continue

        _check_columns(simulation, required=required, label=str(results_path))
        chunk = simulation.loc[
            :, ["repeat", "signal_type", "effect_size", "k", "recall", "fdr"]
        ].copy()
        chunk["method"] = method
        chunk["k"] = chunk["k"].astype(str)
        frames.append(chunk)

    if not frames:
        raise ValueError("No simulation inputs available. Cannot create simulation_summary.parquet.")

    all_sim = pd.concat(frames, ignore_index=True)

    summary = (
        all_sim.groupby(["method", "signal_type", "effect_size", "k"], observed=False)
        .agg(
            recall_mean=("recall", "mean"),
            fdr_mean=("fdr", "mean"),
            recall_sd=("recall", "std"),
            fdr_sd=("fdr", "std"),
            n_repeats=("repeat", "nunique"),
        )
        .reset_index()
        .sort_values(["method", "signal_type", "effect_size", "k"], kind="mergesort")
        .reset_index(drop=True)
    )

    out_path = output_dir / "simulation_summary.parquet"
    summary.to_parquet(out_path, index=False)
    logger.info("Saved simulation summary: %s (rows=%d)", out_path, len(summary))
    return summary, missing_methods


def aggregate_protein_overlap(
    stable_sets: dict[str, set[str]],
    output_dir: Path,
    logger,
) -> pd.DataFrame:
    methods = sorted(stable_sets.keys())
    if not methods:
        raise ValueError("No stable sets available. Cannot create protein_overlap.parquet.")

    rows: list[dict[str, object]] = []
    for method_a, method_b in combinations_with_replacement(methods, 2):
        set_a = stable_sets.get(method_a, set())
        set_b = stable_sets.get(method_b, set())

        n_a = int(len(set_a))
        n_b = int(len(set_b))
        n_intersection = int(len(set_a.intersection(set_b)))
        n_union = int(len(set_a.union(set_b)))
        jaccard = float(n_intersection / n_union) if n_union > 0 else float("nan")

        rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "n_a": n_a,
                "n_b": n_b,
                "n_intersection": n_intersection,
                "n_union": n_union,
                "jaccard": jaccard,
            }
        )

    overlap = pd.DataFrame(rows).sort_values(
        ["method_a", "method_b"], kind="mergesort"
    ).reset_index(drop=True)

    out_path = output_dir / "protein_overlap.parquet"
    overlap.to_parquet(out_path, index=False)
    logger.info("Saved protein overlap: %s (rows=%d)", out_path, len(overlap))
    return overlap


def main() -> None:
    start_time = time.time()
    args = parse_args()

    methods = [str(m).strip() for m in args.methods if str(m).strip()]
    if not methods:
        raise ValueError("--methods cannot be empty.")

    if not 0.0 <= args.pi_threshold <= 1.0:
        raise ValueError(
            f"--pi-threshold must be in [0, 1], got {args.pi_threshold}."
        )

    logger = setup_logging(
        save_results=args.save_results,
        log_subdir="comparison",
        script_name="06_aggregate_results",
    )

    logger.info("Starting 06_aggregate_results.py")
    logger.info(
        "Args: methods=%s results_dir=%s output_dir=%s pi_threshold=%.3f strict_missing=%s",
        methods,
        args.results_dir,
        args.output_dir,
        args.pi_threshold,
        args.strict_missing,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    classifier_summary, missing_classifier = aggregate_classifier_summary(
        methods=methods,
        results_dir=args.results_dir,
        output_dir=output_dir,
        strict_missing=args.strict_missing,
        logger=logger,
    )

    stability_summary, frequency_curve, stable_sets, missing_stability = aggregate_stability_summary(
        methods=methods,
        results_dir=args.results_dir,
        output_dir=output_dir,
        pi_threshold=float(args.pi_threshold),
        strict_missing=args.strict_missing,
        logger=logger,
    )

    enrichment_summary, missing_enrichment = aggregate_enrichment_summary(
        methods=methods,
        results_dir=args.results_dir,
        output_dir=output_dir,
        strict_missing=args.strict_missing,
        logger=logger,
    )

    simulation_summary, missing_simulation = aggregate_simulation_summary(
        methods=methods,
        results_dir=args.results_dir,
        output_dir=output_dir,
        strict_missing=args.strict_missing,
        logger=logger,
    )

    overlap = aggregate_protein_overlap(
        stable_sets=stable_sets,
        output_dir=output_dir,
        logger=logger,
    )

    runtime_seconds = float(time.time() - start_time)

    meta = {
        "methods_requested": methods,
        "methods_present": {
            "classifier": sorted(set(classifier_summary["method"].astype(str).tolist())),
            "stability": sorted(set(stability_summary["method"].astype(str).tolist())),
            "enrichment": sorted(set(enrichment_summary["method"].astype(str).tolist())),
            "simulation": sorted(set(simulation_summary["method"].astype(str).tolist())),
            "overlap": sorted(
                set(overlap["method_a"].astype(str).tolist())
                .union(set(overlap["method_b"].astype(str).tolist()))
            ),
        },
        "missing_methods": {
            "classifier": sorted(set(missing_classifier)),
            "stability": sorted(set(missing_stability)),
            "enrichment": sorted(set(missing_enrichment)),
            "simulation": sorted(set(missing_simulation)),
        },
        "pi_threshold": float(args.pi_threshold),
        "runtime_seconds": runtime_seconds,
        "outputs": {
            "classifier_summary": str(output_dir / "classifier_summary.parquet"),
            "stability_summary": str(output_dir / "stability_summary.parquet"),
            "selection_frequency_curve": str(output_dir / "selection_frequency_curve.parquet"),
            "enrichment_summary": str(output_dir / "enrichment_summary.parquet"),
            "simulation_summary": str(output_dir / "simulation_summary.parquet"),
            "protein_overlap": str(output_dir / "protein_overlap.parquet"),
            "figures_dir": str(output_dir / "figures"),
        },
    }

    meta_path = output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    logger.info("Saved metadata: %s", meta_path)
    logger.info("Finished in %.2f s", runtime_seconds)


if __name__ == "__main__":
    main()
