"""Data preprocessing script.

This script loads raw data from data/raw/clean_dat.csv, filters for two cohorts:
  1. Sepsis=True, ARDS=False
  2. Sepsis=True, ARDS=True, not mild (moderate or severe)

The "not mild" determination is derived from whichever of the columns
ards_severity, ards_mild, ards_notmild contain non-NaN values for a given sample.
Samples lacking information on either Sepsis or ards are dropped.

The combined filtered dataset is then split (stratified on ards) into:
  - seen.csv   — train + validation set
  - unseen.csv — held-out test set

Processed data is saved to data/processed/.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import time

import pandas as pd
from sklearn.model_selection import train_test_split

from src.core.data_utils import filter_data, load_data
from src.core.logging_utils import setup_logging


def split_data(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train+val / test split on the ards column."""
    seen, unseen = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["ards"],
    )
    logger.info(
        "Split — seen: %d rows (%d ARDS), unseen: %d rows (%d ARDS)",
        len(seen), seen["ards"].sum(),
        len(unseen), unseen["ards"].sum(),
    )
    return seen, unseen


def save_outputs(
    combined: pd.DataFrame,
    seen: pd.DataFrame,
    unseen: pd.DataFrame,
    processed_dir: Path,
    test_size: float,
    random_state: int,
    logger: logging.Logger,
) -> None:
    """Save filtered_data.csv, seen.csv, unseen.csv, and split_info.json."""
    processed_dir.mkdir(parents=True, exist_ok=True)

    combined.to_csv(processed_dir / "filtered_data.csv", index=False)
    logger.info("Saved filtered_data.csv  (%d rows)", len(combined))

    seen.to_csv(processed_dir / "seen.csv", index=False)
    logger.info("Saved seen.csv  (%d rows)", len(seen))

    unseen.to_csv(processed_dir / "unseen.csv", index=False)
    logger.info("Saved unseen.csv  (%d rows)", len(unseen))

    split_info = {
        "test_size": test_size,
        "random_state": random_state,
        "n_seen": len(seen),
        "n_unseen": len(unseen),
        "n_ards_seen": int(seen["ards"].sum()),
        "n_ards_unseen": int(unseen["ards"].sum()),
    }
    with open(processed_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    logger.info("Saved split_info.json  %s", split_info)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start = time.time()

    parser = argparse.ArgumentParser(
        description="Filter raw proteomics data into cohorts and produce train/test splits."
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=Path("data/raw/clean_dat.csv"),
        help="Path to the raw input CSV (default: data/raw/clean_dat.csv).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for processed output files (default: data/processed).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of data to hold out as the unseen test set (default: 0.20).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the train/test split (default: 42).",
    )
    args = parser.parse_args()

    logger = setup_logging(save_results=False, script_name="preprocess")

    logger.info("Starting preprocess.py")
    logger.info(
        "Args: raw_path=%s  processed_dir=%s  test_size=%s  random_state=%s",
        args.raw_path, args.processed_dir, args.test_size, args.random_state,
    )

    raw_data = load_data(args.raw_path, logger)
    combined = filter_data(raw_data, logger)
    seen, unseen = split_data(combined, args.test_size, args.random_state, logger)
    save_outputs(
        combined, seen, unseen,
        args.processed_dir, args.test_size, args.random_state,
        logger,
    )

    logger.info("Finished in %.2f s", time.time() - start)


if __name__ == "__main__":
    main()