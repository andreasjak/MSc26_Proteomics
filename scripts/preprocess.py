"""Data preprocessing script.

This script loads raw data from data/raw/clean_dat.csv, filters for two cohorts:
  1. Sepsis=True, ARDS=False
  2. Sepsis=True, ARDS=True, not mild (moderate or severe)

The "not mild" determination is derived from whichever of the columns
ards_severity, ards_mild, ards_notmild contain non-NaN values for a given sample.
Samples lacking information on either Sepsis or ards are dropped.

Processed data is saved to data/processed/.
"""

import logging
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW_PATH = Path("data/raw/clean_dat.csv")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "filtered_data.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

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


def resolve_notmild(df: pd.DataFrame) -> pd.DataFrame:
    """Add a resolved 'not_mild_ards' column; return only rows with a clear verdict."""
    df = df.copy()
    df["_not_mild_ards"] = df.apply(_is_not_mild, axis=1)

    n_no_info = df["_not_mild_ards"].isna().sum()
    if n_no_info:
        log.warning(
            "%d ARDS-positive sample(s) dropped: missing or contradictory "
            "severity information across ards_severity / ards_mild / ards_notmild.",
            n_no_info,
        )

    return df[df["_not_mild_ards"].notna()].copy()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Main preprocessing pipeline."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    log.info("Loading %s", RAW_PATH)
    df = pd.read_csv(RAW_PATH)
    log.info("Loaded %d rows, %d columns", *df.shape)

    # ------------------------------------------------------------------
    # 2. Require non-NaN Sepsis and ards columns — drop anything else
    # ------------------------------------------------------------------
    n_before = len(df)
    df = df.dropna(subset=["Sepsis", "ards"])
    n_dropped = n_before - len(df)
    if n_dropped:
        log.warning(
            "Dropped %d row(s) with missing Sepsis or ards values.", n_dropped
        )
    log.info("%d rows remain after requiring Sepsis and ards values.", len(df))

    # Cast to bool to be safe with 0/1 or True/False encodings
    df["Sepsis"] = df["Sepsis"].astype(bool)
    df["ards"] = df["ards"].astype(bool)

    # ------------------------------------------------------------------
    # 3. Cohort 1 — Sepsis, no ARDS
    # ------------------------------------------------------------------
    cohort1 = df[df["Sepsis"] & ~df["ards"]].copy()
    log.info("Cohort 1 (Sepsis, no ARDS): %d samples", len(cohort1))

    # ------------------------------------------------------------------
    # 4. Cohort 2 — Sepsis + ARDS, severity not mild
    # ------------------------------------------------------------------
    ards_positive = df[df["Sepsis"] & df["ards"]].copy()
    log.info(
        "Sepsis + ARDS (before severity filter): %d samples", len(ards_positive)
    )

    ards_resolved = resolve_notmild(ards_positive)
    cohort2 = ards_resolved[ards_resolved["_not_mild_ards"]].drop(
        columns=["_not_mild_ards"]
    )
    log.info("Cohort 2 (Sepsis + not-mild ARDS): %d samples", len(cohort2))

    # ------------------------------------------------------------------
    # 5. Combine and save
    # ------------------------------------------------------------------
    combined = pd.concat([cohort1, cohort2], ignore_index=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    log.info(
        "Done. Total rows: %d  (Cohort 1: %d, Cohort 2: %d) → saved to %s",
        len(combined), len(cohort1), len(cohort2), OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()