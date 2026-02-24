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

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    """Configure logging to terminal (StreamHandler) only."""
    logger = logging.getLogger("preprocess")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s  %(levelname)s  %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_not_mild(row: pd.Series) -> pd.Series:
    """Return a boolean Series indicating whether each sample is 'not mild' ARDS.

    Uses whichever of ards_severity / ards_mild / ards_notmild is non-NaN.
    If multiple columns are populated and they conflict, the sample is dropped
    (returns NaN so it can be identified downstream).
    """
    votes_notmild = []

    if pd.notna(row.get("ards_severity")):
        votes_notmild.append(str(row["ards_severity"]).strip().lower() != "mild")

    if pd.notna(row.get("ards_mild")):
        votes_notmild.append(not bool(row["ards_mild"]))

    if pd.notna(row.get("ards_notmild")):
        votes_notmild.append(bool(row["ards_notmild"]))

    if not votes_notmild:
        return None  # no severity info available

    if len(set(votes_notmild)) > 1:
        return None  # columns contradict each other

    return votes_notmild[0]


def resolve_notmild(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Add a resolved '_not_mild_ards' column; return only rows with a clear verdict."""
    df = df.copy()
    df["_not_mild_ards"] = df.apply(_is_not_mild, axis=1)

    n_no_info = df["_not_mild_ards"].isna().sum()
    if n_no_info:
        logger.warning(
            "%d ARDS-positive sample(s) dropped: missing or contradictory "
            "severity information across ards_severity / ards_mild / ards_notmild.",
            n_no_info,
        )

    return df[df["_not_mild_ards"].notna()].copy()


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def load_and_filter(raw_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Load raw data and apply cohort filtering logic."""
    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    logger.info("Loading %s", raw_path)
    df = pd.read_csv(raw_path)
    logger.info("Loaded %d rows, %d columns", *df.shape)

    # ------------------------------------------------------------------
    # 2. Require non-NaN Sepsis and ards columns — drop anything else
    # ------------------------------------------------------------------
    n_before = len(df)
    df = df.dropna(subset=["Sepsis", "ards"])
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning(
            "Dropped %d row(s) with missing Sepsis or ards values.", n_dropped
        )
    logger.info("%d rows remain after requiring Sepsis and ards values.", len(df))

    # Cast to bool to be safe with 0/1 or True/False encodings
    df["Sepsis"] = df["Sepsis"].astype(bool)
    df["ards"] = df["ards"].astype(bool)

    # ------------------------------------------------------------------
    # 3. Cohort 1 — Sepsis, no ARDS
    # ------------------------------------------------------------------
    cohort1 = df[df["Sepsis"] & ~df["ards"]].copy()
    logger.info("Cohort 1 (Sepsis, no ARDS): %d samples", len(cohort1))

    # ------------------------------------------------------------------
    # 4. Cohort 2 — Sepsis + ARDS, severity not mild
    # ------------------------------------------------------------------
    ards_positive = df[df["Sepsis"] & df["ards"]].copy()
    logger.info(
        "Sepsis + ARDS (before severity filter): %d samples", len(ards_positive)
    )

    ards_resolved = resolve_notmild(ards_positive, logger)
    cohort2 = ards_resolved[ards_resolved["_not_mild_ards"]].drop(
        columns=["_not_mild_ards"]
    )
    logger.info("Cohort 2 (Sepsis + not-mild ARDS): %d samples", len(cohort2))

    # ------------------------------------------------------------------
    # 5. Combine
    # ------------------------------------------------------------------
    combined = pd.concat([cohort1, cohort2], ignore_index=True)
    logger.info(
        "Combined filtered dataset: %d rows  (Cohort 1: %d, Cohort 2: %d)",
        len(combined), len(cohort1), len(cohort2),
    )
    return combined


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
        default=0.05,
        help="Fraction of data to hold out as the unseen test set (default: 0.05).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the train/test split (default: 42).",
    )
    args = parser.parse_args()

    logger = setup_logging()

    logger.info("Starting preprocess.py")
    logger.info(
        "Args: raw_path=%s  processed_dir=%s  test_size=%s  random_state=%s",
        args.raw_path, args.processed_dir, args.test_size, args.random_state,
    )

    combined = load_and_filter(args.raw_path, logger)
    seen, unseen = split_data(combined, args.test_size, args.random_state, logger)
    save_outputs(
        combined, seen, unseen,
        args.processed_dir, args.test_size, args.random_state,
        logger,
    )

    logger.info("Finished in %.2f s", time.time() - start)


if __name__ == "__main__":
    main()