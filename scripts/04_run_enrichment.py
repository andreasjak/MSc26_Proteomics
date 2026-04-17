"""Run Stage 7 enrichment validation for one selection method."""

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
    BES_CAP_C,
    BES_JACCARD_TAU,
    DATA_PROCESSED,
    ENRICHMENT_LIBRARIES,
    ENRICHMENT_Q_THRESHOLD,
    PERMUTATION_B,
    RANDOM_SEED,
    RESULTS_DIR,
)
from src.enrichment import compute_bes, permutation_null, run_ora
from src.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ORA/BES enrichment validation for a method's stable set.",
    )
    parser.add_argument("--method", type=str, required=True, help="Selection method name.")
    parser.add_argument(
        "--selection-dir",
        type=Path,
        default=RESULTS_DIR / "selection",
        help="Directory containing selection outputs (default: results/selection).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "enrichment",
        help="Directory for enrichment outputs (default: results/enrichment).",
    )
    parser.add_argument(
        "--annotation-path",
        type=Path,
        default=DATA_PROCESSED / "somalogic_annotation.csv",
        help="Path to SomaLogic annotation CSV.",
    )
    parser.add_argument(
        "--protein-ids-path",
        type=Path,
        default=DATA_PROCESSED / "protein_ids.json",
        help="Path to Stage 2 protein_ids.json for measured-protein background.",
    )
    parser.add_argument(
        "--libraries",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit library list. Defaults to ENRICHMENT_LIBRARIES from config.",
    )
    parser.add_argument(
        "--b-perm",
        type=int,
        default=PERMUTATION_B,
        help=f"Number of null permutations (default: {PERMUTATION_B}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Global seed for permutation null sampling (default: {RANDOM_SEED}).",
    )
    parser.add_argument(
        "--skip-null",
        action="store_true",
        help="Skip permutation-null computation and only report raw BES.",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="If set, write logs to file via shared logging utility.",
    )
    return parser.parse_args()


def _normalize_probe_id(value: object) -> str:
    probe = str(value).strip()
    if probe.startswith("seq."):
        probe = probe[4:]
    probe = probe.replace("_", "-")
    probe = probe.replace(".", "-")
    return probe


def _coerce_symbol(value: object) -> str:
    symbol = str(value).strip()
    if not symbol or symbol.lower() == "nan":
        return ""
    return symbol


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _load_stable_set(stable_set_path: Path) -> dict:
    if not stable_set_path.exists():
        raise FileNotFoundError(
            f"Stable set file not found at {stable_set_path}. Run scripts/02_run_selection.py first."
        )
    return json.loads(stable_set_path.read_text(encoding="utf-8"))


def _load_annotation(annotation_path: Path) -> pd.DataFrame:
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    ann = pd.read_csv(annotation_path).copy()
    if "PROBEID" not in ann.columns or "SYMBOL" not in ann.columns:
        raise ValueError("Annotation CSV must contain PROBEID and SYMBOL columns.")
    ann["probe_norm"] = ann["PROBEID"].map(_normalize_probe_id)
    ann["symbol_clean"] = ann["SYMBOL"].map(_coerce_symbol)
    ann = ann.drop_duplicates(subset=["probe_norm"], keep="first")
    return ann


def _make_probe_to_symbol(annotation_df: pd.DataFrame) -> dict[str, str]:
    return dict(
        zip(
            annotation_df["probe_norm"].astype(str),
            annotation_df["symbol_clean"].astype(str),
        )
    )


def _load_background_proteins(protein_ids_path: Path, annotation_df: pd.DataFrame) -> list[str]:
    if protein_ids_path.exists():
        return json.loads(protein_ids_path.read_text(encoding="utf-8"))
    return annotation_df["probe_norm"].astype(str).tolist()


def _map_proteins_to_genes(protein_ids: list[str], probe_to_symbol: dict[str, str]) -> tuple[list[str], int]:
    mapped: list[str] = []
    missing = 0
    for protein_id in protein_ids:
        probe_norm = _normalize_probe_id(protein_id)
        symbol = _coerce_symbol(probe_to_symbol.get(probe_norm, ""))
        if symbol:
            mapped.append(symbol)
        else:
            missing += 1
    return _unique_preserve_order(mapped), missing


def _null_cache_path(base_output_dir: Path, library: str, size: int, seed: int) -> Path:
    return base_output_dir / "_null_cache" / library / f"size_{size}_seed_{seed}.npz"


def _null_stats(bes_raw: float, null_values: np.ndarray) -> tuple[float, float, int]:
    valid = null_values[np.isfinite(null_values)]
    if valid.size == 0:
        return float("nan"), float("nan"), 0

    mean = float(valid.mean())
    std = float(valid.std(ddof=0))
    bes_z = float((bes_raw - mean) / std) if std > 0 else float("nan")
    bes_p_emp = float((np.sum(valid >= bes_raw) + 1) / (len(valid) + 1))
    return bes_z, bes_p_emp, int(len(valid))


def main() -> None:
    start_time = time.time()
    args = parse_args()

    if args.b_perm <= 0:
        raise ValueError(f"--b-perm must be > 0, got {args.b_perm}.")

    logger = setup_logging(
        save_results=args.save_results,
        log_subdir=f"enrichment/{args.method}",
        script_name="04_run_enrichment",
    )

    libraries = args.libraries if args.libraries else ENRICHMENT_LIBRARIES

    logger.info("Starting 04_run_enrichment.py")
    logger.info(
        "Args: method=%s selection_dir=%s output_dir=%s annotation_path=%s protein_ids_path=%s skip_null=%s b_perm=%d seed=%d",
        args.method,
        args.selection_dir,
        args.output_dir,
        args.annotation_path,
        args.protein_ids_path,
        args.skip_null,
        args.b_perm,
        args.seed,
    )

    method_selection_dir = args.selection_dir / args.method
    stable_set_path = method_selection_dir / "stable_set.json"
    stable_payload = _load_stable_set(stable_set_path)

    stable_proteins = [str(p) for p in stable_payload.get("protein_ids", [])]
    if not stable_proteins:
        raise ValueError(f"Stable set is empty for method '{args.method}'.")

    annotation_df = _load_annotation(args.annotation_path)
    probe_to_symbol = _make_probe_to_symbol(annotation_df)

    gene_list, missing_stable = _map_proteins_to_genes(stable_proteins, probe_to_symbol)
    if not gene_list:
        raise ValueError(
            "No stable proteins could be mapped to gene symbols. "
            "Check stable_set.json and annotation mapping."
        )

    background_proteins = _load_background_proteins(args.protein_ids_path, annotation_df)
    background_genes, missing_background = _map_proteins_to_genes(background_proteins, probe_to_symbol)

    if not background_genes:
        raise ValueError("Background gene list is empty after annotation mapping.")

    background_set = set(background_genes)
    gene_list = [g for g in gene_list if g in background_set]
    if not gene_list:
        raise ValueError("Gene list became empty after intersecting with background.")

    logger.info(
        "Mapped stable proteins to genes: %d -> %d genes (missing=%d)",
        len(stable_proteins),
        len(gene_list),
        missing_stable,
    )
    logger.info(
        "Mapped background proteins to genes: %d -> %d genes (missing=%d)",
        len(background_proteins),
        len(background_genes),
        missing_background,
    )

    method_out_dir = args.output_dir / args.method
    method_out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []

    for library in libraries:
        logger.info("Running library: %s", library)
        library_out_dir = method_out_dir / library
        library_out_dir.mkdir(parents=True, exist_ok=True)

        terms_path = library_out_dir / "terms.parquet"
        bes_components_path = library_out_dir / "bes_components.json"

        term_df = run_ora(
            gene_list=gene_list,
            background=background_genes,
            library=library,
        )
        term_df.to_parquet(terms_path, index=False)

        bes_info = compute_bes(
            term_df=term_df,
            gene_list=gene_list,
            c=BES_CAP_C,
            tau=BES_JACCARD_TAU,
            q_threshold=ENRICHMENT_Q_THRESHOLD,
        )

        bes_raw = float(bes_info["bes_raw"])
        bes_z = float("nan")
        bes_p_emp = float("nan")
        null_effective = 0

        if not args.skip_null:
            cache_path = _null_cache_path(
                base_output_dir=args.output_dir,
                library=library,
                size=len(gene_list),
                seed=args.seed,
            )
            if cache_path.exists():
                null_values = np.load(cache_path)["null_values"]
                logger.info("Loaded null cache: %s", cache_path)
            else:
                logger.info(
                    "Computing permutation null: library=%s size=%d b_perm=%d",
                    library,
                    len(gene_list),
                    args.b_perm,
                )
                null_values = permutation_null(
                    background=background_genes,
                    gene_list_size=len(gene_list),
                    library=library,
                    b_perm=args.b_perm,
                    c=BES_CAP_C,
                    tau=BES_JACCARD_TAU,
                    q_threshold=ENRICHMENT_Q_THRESHOLD,
                    seed=args.seed,
                )
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cache_path,
                    null_values=null_values,
                    gene_list_size=np.array([len(gene_list)], dtype=np.int64),
                    seed=np.array([args.seed], dtype=np.int64),
                )
                logger.info("Saved null cache: %s", cache_path)

            bes_z, bes_p_emp, null_effective = _null_stats(bes_raw=bes_raw, null_values=null_values)

        bes_payload = {
            "library": library,
            "method": args.method,
            "gene_list_size": int(len(gene_list)),
            "background_size": int(len(background_genes)),
            "bes_raw": bes_raw,
            "bes_z": bes_z,
            "bes_p_emp": bes_p_emp,
            "null_effective": int(null_effective),
            "skip_null": bool(args.skip_null),
            "bes_components": bes_info,
        }
        with bes_components_path.open("w", encoding="utf-8") as f:
            json.dump(bes_payload, f, indent=2)

        summary_rows.append(
            {
                "library": library,
                "gene_list_size": int(len(gene_list)),
                "bes_raw": bes_raw,
                "bes_z": bes_z,
                "bes_p_emp": bes_p_emp,
                "n_significant_terms": int(bes_info["n_significant_terms"]),
            }
        )

        logger.info(
            "Library done: %s | BES=%.4f z=%s p_emp=%s significant_terms=%d",
            library,
            bes_raw,
            f"{bes_z:.3f}" if np.isfinite(bes_z) else "nan",
            f"{bes_p_emp:.4f}" if np.isfinite(bes_p_emp) else "nan",
            int(bes_info["n_significant_terms"]),
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = method_out_dir / "summary.parquet"
    summary_df.to_parquet(summary_path, index=False)

    meta = {
        "method": args.method,
        "libraries": list(libraries),
        "selection_dir": str(args.selection_dir),
        "output_dir": str(args.output_dir),
        "annotation_path": str(args.annotation_path),
        "protein_ids_path": str(args.protein_ids_path),
        "skip_null": bool(args.skip_null),
        "b_perm": int(args.b_perm),
        "seed": int(args.seed),
        "gene_list_size": int(len(gene_list)),
        "background_size": int(len(background_genes)),
        "summary_path": str(summary_path),
        "runtime_seconds": float(time.time() - start_time),
    }
    meta_path = method_out_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved summary: %s", summary_path)
    logger.info("Saved metadata: %s", meta_path)
    logger.info("Finished in %.2f s", time.time() - start_time)


if __name__ == "__main__":
    main()
