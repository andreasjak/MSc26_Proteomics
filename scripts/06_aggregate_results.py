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
from src.statistics import nadeau_bengio_se


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
        "--splits-meta",
        type=Path,
        default=Path("data/processed/splits_meta.json"),
        help="Path to splits_meta.json for Nadeau-Bengio n_train/n_test (default: data/processed/splits_meta.json).",
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
    n_train: int,
    n_test: int,
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

    summary["auc_se_nb"] = summary.apply(
        lambda r: nadeau_bengio_se(r["auc_sd"], int(r["n_splits"]), n_train, n_test),
        axis=1,
    )
    summary["aupr_se_nb"] = summary.apply(
        lambda r: nadeau_bengio_se(r["aupr_sd"], int(r["n_splits"]), n_train, n_test),
        axis=1,
    )
    summary["n_train"] = int(n_train)
    summary["n_test"] = int(n_test)

    out_path = output_dir / "classifier_summary.parquet"
    summary.to_parquet(out_path, index=False)
    logger.info("Saved classifier summary: %s (rows=%d)", out_path, len(summary))
    return summary, missing_methods


def _mean_pairwise_jaccard_topk(
    selections: pd.DataFrame,
    k: int,
) -> float:
    if not {"split_id", "rank", "protein_id"}.issubset(selections.columns):
        return float("nan")

    top = selections.loc[selections["rank"] < k, ["split_id", "protein_id"]]
    sets_by_split: list[set[str]] = [
        set(group["protein_id"].astype(str).tolist())
        for _, group in top.groupby("split_id", sort=True)
    ]
    if len(sets_by_split) < 2:
        return float("nan")

    jaccards: list[float] = []
    for i in range(len(sets_by_split)):
        for j in range(i + 1, len(sets_by_split)):
            a, b = sets_by_split[i], sets_by_split[j]
            union = a | b
            if not union:
                continue
            jaccards.append(len(a & b) / len(union))

    return float(np.mean(jaccards)) if jaccards else float("nan")


def aggregate_stability_summary(
    methods: list[str],
    results_dir: Path,
    output_dir: Path,
    pi_threshold: float,
    strict_missing: bool,
    logger,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    rows: list[dict[str, object]] = []
    curve_frames: list[pd.DataFrame] = []
    missing_methods: list[str] = []

    required = {"protein_id", "frequency"}

    for method in methods:
        stability_path = results_dir / "selection" / method / "stability_significant.parquet"
        selections_path = results_dir / "selection" / method / "selections.parquet"

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

        selections = _load_parquet(
            path=selections_path,
            label=f"selections for method='{method}'",
            strict_missing=strict_missing,
            logger=logger,
        )
        mean_jaccard_top50 = (
            _mean_pairwise_jaccard_topk(selections, k=50)
            if selections is not None
            else float("nan")
        )

        rows.append(
            {
                "method": method,
                "stable_size_pi_0_3": stable_size_pi,
                "top_frequency": top_frequency,
                "n_ge_0_5": n_ge_05,
                "n_ge_0_3": n_ge_03,
                "mean_jaccard_top50": mean_jaccard_top50,
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

    return summary, frequency_curve, missing_methods


def aggregate_full_data_summary(
    methods: list[str],
    results_dir: Path,
    output_dir: Path,
    strict_missing: bool,
    logger,
) -> tuple[pd.DataFrame, dict[str, set[str]], list[str]]:
    rows: list[dict[str, object]] = []
    full_data_sets: dict[str, set[str]] = {}
    missing_methods: list[str] = []

    required = {"rank", "protein_id", "score", "significant"}

    for method in methods:
        ranking_path = results_dir / "selection" / method / "full_data_ranking.parquet"
        ranking = _load_parquet(
            path=ranking_path,
            label=f"full_data_ranking for method='{method}'",
            strict_missing=strict_missing,
            logger=logger,
        )
        if ranking is None:
            missing_methods.append(method)
            continue

        _check_columns(ranking, required=required, label=str(ranking_path))
        ranking = ranking.copy()
        ranking["protein_id"] = ranking["protein_id"].astype(str)
        ranking["significant"] = ranking["significant"].astype(bool)
        ranking["rank"] = pd.to_numeric(ranking["rank"], errors="coerce").astype("Int64")

        sig_ids = set(ranking.loc[ranking["significant"], "protein_id"].tolist())
        full_data_sets[method] = sig_ids

        top_row = ranking.sort_values("rank", kind="mergesort").iloc[0]
        rows.append(
            {
                "method": method,
                "n_significant": int(len(sig_ids)),
                "top_protein": str(top_row["protein_id"]),
                "top_score": float(top_row["score"]),
            }
        )

    if not rows:
        raise ValueError(
            "No full-data inputs available. Cannot create full_data_summary.parquet."
        )

    summary = pd.DataFrame(rows).sort_values(["method"], kind="mergesort").reset_index(drop=True)
    out_path = output_dir / "full_data_summary.parquet"
    summary.to_parquet(out_path, index=False)
    logger.info("Saved full-data summary: %s (rows=%d)", out_path, len(summary))

    return summary, full_data_sets, missing_methods


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
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    recall_frames: list[pd.DataFrame] = []
    fdr_frames: list[pd.DataFrame] = []
    missing_methods: list[str] = []
    recall_required = {"repeat", "signal_type", "effect_size", "k", "recall"}
    fdr_required = {"repeat", "k", "fdr", "n_selected"}

    for method in methods:
        method_dir = results_dir / "simulation" / method
        recall_path = method_dir / "recall.parquet"
        fdr_path = method_dir / "fdr.parquet"

        recall = _load_parquet(
            path=recall_path,
            label=f"simulation recall for method='{method}'",
            strict_missing=strict_missing,
            logger=logger,
        )
        fdr = _load_parquet(
            path=fdr_path,
            label=f"simulation fdr for method='{method}'",
            strict_missing=strict_missing,
            logger=logger,
        )
        if recall is None or fdr is None:
            missing_methods.append(method)
            continue

        _check_columns(recall, required=recall_required, label=str(recall_path))
        _check_columns(fdr, required=fdr_required, label=str(fdr_path))

        recall_chunk = recall.loc[
            :, ["repeat", "signal_type", "effect_size", "k", "recall"]
        ].copy()
        recall_chunk["method"] = method
        recall_chunk["k"] = recall_chunk["k"].astype(str)
        recall_frames.append(recall_chunk)

        fdr_chunk = fdr.loc[:, ["repeat", "k", "fdr", "n_selected"]].copy()
        fdr_chunk["method"] = method
        fdr_chunk["k"] = fdr_chunk["k"].astype(str)
        fdr_frames.append(fdr_chunk)

    if not recall_frames or not fdr_frames:
        raise ValueError(
            "No simulation inputs available. Cannot create recall_summary.parquet / fdr_summary.parquet."
        )

    all_recall = pd.concat(recall_frames, ignore_index=True)
    all_fdr = pd.concat(fdr_frames, ignore_index=True)

    recall_summary = (
        all_recall.groupby(["method", "signal_type", "effect_size", "k"], observed=False)
        .agg(
            recall_mean=("recall", "mean"),
            recall_sd=("recall", "std"),
            n_repeats=("repeat", "nunique"),
        )
        .reset_index()
        .sort_values(["method", "signal_type", "effect_size", "k"], kind="mergesort")
        .reset_index(drop=True)
    )

    fdr_summary = (
        all_fdr.groupby(["method", "k"], observed=False)
        .agg(
            fdr_mean=("fdr", "mean"),
            fdr_sd=("fdr", "std"),
            n_repeats=("repeat", "nunique"),
            n_selected_mean=("n_selected", "mean"),
        )
        .reset_index()
        .sort_values(["method", "k"], kind="mergesort")
        .reset_index(drop=True)
    )

    recall_out = output_dir / "recall_summary.parquet"
    fdr_out = output_dir / "fdr_summary.parquet"
    recall_summary.to_parquet(recall_out, index=False)
    fdr_summary.to_parquet(fdr_out, index=False)
    logger.info("Saved recall summary: %s (rows=%d)", recall_out, len(recall_summary))
    logger.info("Saved FDR summary: %s (rows=%d)", fdr_out, len(fdr_summary))
    return recall_summary, fdr_summary, missing_methods


def aggregate_protein_overlap(
    protein_sets: dict[str, set[str]],
    output_dir: Path,
    logger,
    filename: str = "protein_overlap.parquet",
    label: str = "protein overlap",
) -> pd.DataFrame:
    methods = sorted(protein_sets.keys())
    if not methods:
        raise ValueError(f"No protein sets available. Cannot create {filename}.")

    rows: list[dict[str, object]] = []
    for method_a, method_b in combinations_with_replacement(methods, 2):
        set_a = protein_sets.get(method_a, set())
        set_b = protein_sets.get(method_b, set())

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

    out_path = output_dir / filename
    overlap.to_parquet(out_path, index=False)
    logger.info("Saved %s: %s (rows=%d)", label, out_path, len(overlap))
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

    splits_meta = _load_json(args.splits_meta, strict_missing=True, logger=logger)
    test_size = float(splits_meta["test_size"])
    n_samples = int(splits_meta["n_samples"])
    n_test = int(round(test_size * n_samples))
    n_train = n_samples - n_test
    logger.info(
        "Nadeau-Bengio: n_samples=%d test_size=%.4f -> n_train=%d n_test=%d (ratio=%.4f)",
        n_samples,
        test_size,
        n_train,
        n_test,
        n_test / n_train,
    )

    classifier_summary, missing_classifier = aggregate_classifier_summary(
        methods=methods,
        results_dir=args.results_dir,
        output_dir=output_dir,
        strict_missing=args.strict_missing,
        logger=logger,
        n_train=n_train,
        n_test=n_test,
    )

    stability_summary, frequency_curve, missing_stability = aggregate_stability_summary(
        methods=methods,
        results_dir=args.results_dir,
        output_dir=output_dir,
        pi_threshold=float(args.pi_threshold),
        strict_missing=args.strict_missing,
        logger=logger,
    )

    full_data_summary, full_data_sets, missing_full_data = aggregate_full_data_summary(
        methods=methods,
        results_dir=args.results_dir,
        output_dir=output_dir,
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

    recall_summary, fdr_summary, missing_simulation = aggregate_simulation_summary(
        methods=methods,
        results_dir=args.results_dir,
        output_dir=output_dir,
        strict_missing=args.strict_missing,
        logger=logger,
    )

    overlap = aggregate_protein_overlap(
        protein_sets=full_data_sets,
        output_dir=output_dir,
        logger=logger,
        filename="full_data_overlap.parquet",
        label="full-data overlap",
    )

    runtime_seconds = float(time.time() - start_time)

    meta = {
        "methods_requested": methods,
        "methods_present": {
            "classifier": sorted(set(classifier_summary["method"].astype(str).tolist())),
            "stability": sorted(set(stability_summary["method"].astype(str).tolist())),
            "full_data": sorted(set(full_data_summary["method"].astype(str).tolist())),
            "enrichment": sorted(set(enrichment_summary["method"].astype(str).tolist())),
            "simulation": sorted(
                set(recall_summary["method"].astype(str).tolist())
                .intersection(set(fdr_summary["method"].astype(str).tolist()))
            ),
            "overlap": sorted(
                set(overlap["method_a"].astype(str).tolist())
                .union(set(overlap["method_b"].astype(str).tolist()))
            ),
        },
        "missing_methods": {
            "classifier": sorted(set(missing_classifier)),
            "stability": sorted(set(missing_stability)),
            "full_data": sorted(set(missing_full_data)),
            "enrichment": sorted(set(missing_enrichment)),
            "simulation": sorted(set(missing_simulation)),
        },
        "pi_threshold": float(args.pi_threshold),
        "nadeau_bengio": {
            "n_train": int(n_train),
            "n_test": int(n_test),
            "test_train_ratio": float(n_test / n_train),
        },
        "runtime_seconds": runtime_seconds,
        "outputs": {
            "classifier_summary": str(output_dir / "classifier_summary.parquet"),
            "stability_summary": str(output_dir / "stability_summary.parquet"),
            "selection_frequency_curve": str(output_dir / "selection_frequency_curve.parquet"),
            "full_data_summary": str(output_dir / "full_data_summary.parquet"),
            "full_data_overlap": str(output_dir / "full_data_overlap.parquet"),
            "enrichment_summary": str(output_dir / "enrichment_summary.parquet"),
            "recall_summary": str(output_dir / "recall_summary.parquet"),
            "fdr_summary": str(output_dir / "fdr_summary.parquet"),
            "figures_dir": str(output_dir / "figures"),
        },
    }

    meta_path = output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    logger.info("Saved metadata: %s", meta_path)
    logger.info("Finished in %.2f s", runtime_seconds)


if __name__ == "__main__":
    main()
