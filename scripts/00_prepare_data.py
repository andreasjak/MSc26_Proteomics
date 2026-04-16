"""Prepare filtered analysis cohort for Stage 2.

Reads raw data, applies cohort filtering, and writes:
- data/processed/cohort.parquet
- data/processed/protein_ids.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import DATA_PROCESSED, DATA_RAW, RANDOM_SEED
from src.logging_utils import setup_logging
from src.data_loading import load_cohort_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Stage 2 cohort from raw SomaScan/clinical data.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_RAW / "clean_dat.csv",
        help="Path to raw cohort CSV (default: data/raw/clean_dat.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_PROCESSED,
        help="Output directory for cohort artifacts (default: data/processed).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Global seed (logged for reproducibility).",
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
        log_subdir="prepare_data",
        script_name="00_prepare_data",
    )

    logger.info("Starting 00_prepare_data.py")
    logger.info(
        "Args: data_path=%s output_dir=%s seed=%d",
        args.data_path,
        args.output_dir,
        args.seed,
    )
    logger.info(
        "Assumption: SomaScan protein values are already normalized upstream; "
        "no value transformation is applied here."
    )

    cohort_table, protein_ids, audit = load_cohort_table(args.data_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort_out = args.output_dir / "cohort.parquet"
    proteins_out = args.output_dir / "protein_ids.json"

    cohort_table.to_parquet(cohort_out, index=False)
    with proteins_out.open("w", encoding="utf-8") as f:
        json.dump(protein_ids, f, indent=2)

    n_patients = len(cohort_table)
    n_proteins = len(protein_ids)
    n_ards = int(cohort_table["y"].sum())
    prevalence = (n_ards / n_patients) if n_patients else 0.0

    logger.info("Saved cohort table: %s", cohort_out)
    logger.info("Saved protein IDs: %s", proteins_out)
    logger.info("Patients: %d", n_patients)
    logger.info("Proteins: %d", n_proteins)
    logger.info("ARDS positives: %d (%.2f%%)", n_ards, 100.0 * prevalence)
    logger.info("Patient ID source: %s", audit.get("patient_id_source", "unknown"))

    dropped_reasons = [
        ("missing Sepsis or ARDS label", int(audit.get("dropped_missing_sepsis_or_ards", 0))),
        ("non-sepsis", int(audit.get("dropped_non_sepsis", 0))),
        ("mild ARDS", int(audit.get("dropped_mild_ards", 0))),
        (
            "missing/contradictory ARDS severity among ARDS-positive rows",
            int(audit.get("dropped_missing_or_conflicting_severity", 0)),
        ),
    ]

    for reason, count in dropped_reasons:
        if count > 0:
            logger.warning("Dropped %d patient(s): %s", count, reason)

    logger.info(
        "Cohort composition: sepsis-no-ARDS=%d, sepsis-not-mild-ARDS=%d",
        int(audit.get("kept_sepsis_no_ards", 0)),
        int(audit.get("kept_sepsis_not_mild_ards", 0)),
    )
    logger.info("Finished in %.2f s", time.time() - start)


if __name__ == "__main__":
    main()
