"""
data_utils.py
-------------
Reusable data-handling utilities for the MSc26 proteomics project.

The project operates on `filtered_data.csv`, which already contains two cohorts:
  - Sepsis + Not ARDS  (ards == False)
  - Sepsis + Moderate/Severe ARDS  (ards == True, mild ARDS excluded)

Splitting on the `ards` column is therefore sufficient for all downstream analyses.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Data subsets
# ---------------------------------------------------------------------------

class Subsets(NamedTuple):
    """Container returned by :func:`create_subsets`."""
    ards: pd.DataFrame
    non_ards: pd.DataFrame


def create_subsets(df: pd.DataFrame) -> Subsets:
    """Split the filtered dataset into ARDS and Non-ARDS cohorts.

    Parameters
    ----------
    df : pd.DataFrame
        The main dataframe loaded from ``filtered_data.csv``.  It is expected
        to already contain only sepsis patients with mild ARDS removed, so a
        simple boolean split on the ``ards`` column is sufficient.

    Returns
    -------
    Subsets
        A named tuple with fields ``ards`` and ``non_ards``, each a
        ``pd.DataFrame`` view of the corresponding cohort.

    Examples
    --------
    >>> subsets = create_subsets(df)
    >>> subsets.ards.shape, subsets.non_ards.shape
    """
    ards_data = df[df["ards"] == True].copy()
    non_ards_data = df[df["ards"] == False].copy()
    return Subsets(ards=ards_data, non_ards=non_ards_data)


# ---------------------------------------------------------------------------
# Protein feature helpers
# ---------------------------------------------------------------------------

def get_protein_features(df: pd.DataFrame, prefix: str = "seq.") -> list[str]:
    """Return column names that correspond to protein (SomaScan) features.

    Parameters
    ----------
    df : pd.DataFrame
        Any dataframe that contains protein columns.
    prefix : str, optional
        Column-name prefix that identifies protein columns.  Defaults to
        ``"seq."`` (SomaScan convention).

    Returns
    -------
    list[str]
        Sorted list of matching column names.

    Raises
    ------
    AssertionError
        If no columns with the given prefix are found.
    """
    features = [c for c in df.columns if c.startswith(prefix)]
    assert len(features) > 0, (
        f"No columns starting with '{prefix}' found. "
        "Check that the correct dataframe is passed."
    )
    return features


def get_top_diff_features(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    k: int = 6,
    prefix: str = "seq.",
) -> list[str]:
    """Identify proteins with the largest absolute mean difference between two groups.

    Parameters
    ----------
    df_a, df_b : pd.DataFrame
        Two cohort dataframes (e.g. ARDS vs Non-ARDS).  Both must share the
        same protein columns.
    k : int, optional
        Number of top features to return.  Defaults to 6.
    prefix : str, optional
        Column prefix for protein features.  Defaults to ``"seq."``.

    Returns
    -------
    list[str]
        Column names of the *k* proteins with the highest |mean(A) - mean(B)|,
        ordered from largest to smallest difference.
    """
    features = get_protein_features(df_a, prefix=prefix)
    k = min(k, len(features))
    means_a = df_a[features].mean()
    means_b = df_b[features].mean()
    diff = (means_a - means_b).abs()
    return diff.sort_values(ascending=False).head(k).index.tolist()


# ---------------------------------------------------------------------------
# Generic data-loading helpers
# ---------------------------------------------------------------------------

def load_data(
    path: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Load a CSV file and log its shape and ARDS class counts.

    Works for any of the project's tabular data files (``seen.csv``,
    ``unseen.csv``, ``filtered_data.csv``, etc.).

    Parameters
    ----------
    path : Path
        Path to the CSV file.
    logger : logging.Logger
        Logger instance for status messages.

    Returns
    -------
    pd.DataFrame
        The loaded data.
    """
    logger.info("Loading data from %s", path)
    data = pd.read_csv(path)
    logger.info("Data shape: %s", data.shape)

    if "ards" in data.columns:
        n_ards = int((data["ards"] == 1).sum())
        n_non = int((data["ards"] == 0).sum())
        logger.info("Samples — ARDS: %d | non-ARDS: %d", n_ards, n_non)

    return data


def _is_not_mild(row: pd.Series) -> bool | None:
    """Resolve a row's ARDS severity into a not-mild verdict.

    Uses whichever of ``ards_severity``, ``ards_mild``, and ``ards_notmild``
    is present. If multiple populated fields conflict, returns ``None`` so the
    row can be excluded upstream.
    """
    votes_notmild = []

    if pd.notna(row.get("ards_severity")):
        votes_notmild.append(str(row["ards_severity"]).strip().lower() != "mild")

    if pd.notna(row.get("ards_mild")):
        votes_notmild.append(not bool(row["ards_mild"]))

    if pd.notna(row.get("ards_notmild")):
        votes_notmild.append(bool(row["ards_notmild"]))

    if not votes_notmild:
        return None

    if len(set(votes_notmild)) > 1:
        return None

    return votes_notmild[0]


def _resolve_notmild(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Add and validate `_not_mild_ards` for ARDS-positive rows."""
    resolved = df.copy()
    resolved["_not_mild_ards"] = resolved.apply(_is_not_mild, axis=1)

    n_no_info = resolved["_not_mild_ards"].isna().sum()
    if n_no_info:
        logger.warning(
            "%d ARDS-positive sample(s) dropped: missing or contradictory "
            "severity information across ards_severity / ards_mild / ards_notmild.",
            n_no_info,
        )

    return resolved[resolved["_not_mild_ards"].notna()].copy()


def filter_data(
    df: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Filter raw data into the two analysis cohorts.

    Cohort definitions:
      1) Sepsis=True, ARDS=False
      2) Sepsis=True, ARDS=True, and severity not mild
    """
    filtered = df.copy()

    n_before = len(filtered)
    filtered = filtered.dropna(subset=["Sepsis", "ards"])
    n_dropped = n_before - len(filtered)
    if n_dropped:
        logger.warning(
            "Dropped %d row(s) with missing Sepsis or ards values.", n_dropped
        )
    logger.info(
        "%d rows remain after requiring Sepsis and ards values.", len(filtered)
    )

    filtered["Sepsis"] = filtered["Sepsis"].astype(bool)
    filtered["ards"] = filtered["ards"].astype(bool)

    cohort1 = filtered[filtered["Sepsis"] & ~filtered["ards"]].copy()
    logger.info("Cohort 1 (Sepsis, no ARDS): %d samples", len(cohort1))

    ards_positive = filtered[filtered["Sepsis"] & filtered["ards"]].copy()
    logger.info(
        "Sepsis + ARDS (before severity filter): %d samples", len(ards_positive)
    )

    ards_resolved = _resolve_notmild(ards_positive, logger)
    cohort2 = ards_resolved[ards_resolved["_not_mild_ards"]].drop(
        columns=["_not_mild_ards"]
    )
    logger.info("Cohort 2 (Sepsis + not-mild ARDS): %d samples", len(cohort2))

    combined = pd.concat([cohort1, cohort2], ignore_index=True)
    logger.info(
        "Combined filtered dataset: %d rows  (Cohort 1: %d, Cohort 2: %d)",
        len(combined), len(cohort1), len(cohort2),
    )
    return combined


def load_annotation(
    path: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Load the SomaLogic annotation table.

    Cleans the string columns (``PROBEID``, ``SYMBOL``, ``UNIPROT``,
    ``GENENAME``) by replacing ``"nan"`` with empty strings.  If the file
    is not found, returns an empty ``DataFrame`` with the expected columns
    and logs a warning.

    Parameters
    ----------
    path : Path
        Path to the annotation CSV.
    logger : logging.Logger
        Logger instance for status messages.

    Returns
    -------
    pd.DataFrame
        The annotation table.
    """
    try:
        annot = pd.read_csv(path)
        for col in ["PROBEID", "SYMBOL", "UNIPROT", "GENENAME"]:
            if col in annot.columns:
                annot[col] = annot[col].astype(str).replace({"nan": ""})
        logger.info("Annotation loaded: %d rows", len(annot))
    except FileNotFoundError:
        logger.warning(
            "Annotation file not found: %s — proceeding without symbols.", path
        )
        annot = pd.DataFrame(columns=["PROBEID", "SYMBOL", "UNIPROT", "GENENAME"])

    return annot


def load_features(
    path: Path,
    logger: logging.Logger,
) -> list[str]:
    """Load a feature-list CSV and return the protein names.

    The CSV must have a single column with header ``"protein"``.

    Parameters
    ----------
    path : Path
        Path to the features CSV.
    logger : logging.Logger
        Logger instance for status messages.

    Returns
    -------
    list[str]
        Protein feature names.
    """
    logger.info("Loading feature list from %s", path)
    features_df = pd.read_csv(path)
    features = features_df["protein"].tolist()
    logger.info("Features loaded: %d", len(features))
    return features