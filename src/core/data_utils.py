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