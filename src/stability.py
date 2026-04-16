"""Stability metrics and stable-set construction utilities."""

from __future__ import annotations

import pandas as pd


REQUIRED_SELECTION_COLUMNS = {
    "split_id",
    "rank",
    "protein_idx",
    "protein_id",
    "score",
    "significant",
}


def compute_stability(
    selections: pd.DataFrame,
    n_splits: int,
    mode: str = "significant",
    topk: int | None = None,
) -> pd.DataFrame:
    """Compute per-protein stability statistics across splits.

    Parameters
    ----------
    selections : pd.DataFrame
        Long-format dataframe with columns
        [split_id, rank, protein_idx, protein_id, score, significant].
    n_splits : int
        Total number of splits used as denominator for frequency.
    mode : str, optional
        Frequency definition. "significant" uses the significant flag, while
        "topk" uses membership in rank < topk.
    topk : int | None, optional
        Top-k threshold used when mode="topk".

    Returns
    -------
    pd.DataFrame
        Columns [protein_idx, protein_id, frequency, mean_rank, mean_score].
    """
    if n_splits <= 0:
        raise ValueError(f"n_splits must be > 0, got {n_splits}.")

    missing_cols = REQUIRED_SELECTION_COLUMNS.difference(selections.columns)
    if missing_cols:
        raise ValueError(
            "Selections dataframe is missing required columns: "
            f"{sorted(missing_cols)}"
        )

    df = selections.loc[
        :, ["protein_idx", "protein_id", "rank", "score", "significant"]
    ].copy()

    if mode == "significant":
        selected = df["significant"].astype(bool)
    elif mode == "topk":
        if topk is None:
            raise ValueError("topk must be provided when mode='topk'.")
        if topk <= 0:
            raise ValueError(f"topk must be > 0, got {topk}.")
        selected = df["rank"].astype(int) < int(topk)
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Use 'significant' or 'topk'.")

    summary = (
        df.assign(_selected=selected.astype(int))
        .groupby(["protein_idx", "protein_id"], as_index=False)
        .agg(
            selected_count=("_selected", "sum"),
            mean_rank=("rank", "mean"),
            mean_score=("score", "mean"),
        )
    )

    summary["frequency"] = summary["selected_count"] / float(n_splits)
    summary = summary.drop(columns=["selected_count"])
    summary = summary.loc[
        :, ["protein_idx", "protein_id", "frequency", "mean_rank", "mean_score"]
    ]

    summary = summary.sort_values(
        by=["frequency", "mean_rank", "protein_idx"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return summary


def stable_set(
    stability_df: pd.DataFrame,
    pi: float,
    min_size: int,
    topk_fallback: int,
) -> tuple[list[str], str]:
    """Build stable set from stability frequencies.

    Returns (protein_ids, method_used), where method_used is "threshold" or
    "topk_fallback".
    """
    if not 0.0 <= pi <= 1.0:
        raise ValueError(f"pi must be in [0, 1], got {pi}.")
    if min_size < 0:
        raise ValueError(f"min_size must be >= 0, got {min_size}.")
    if topk_fallback < 0:
        raise ValueError(f"topk_fallback must be >= 0, got {topk_fallback}.")

    required_cols = {"protein_idx", "protein_id", "frequency", "mean_rank"}
    missing_cols = required_cols.difference(stability_df.columns)
    if missing_cols:
        raise ValueError(
            "stability_df is missing required columns: "
            f"{sorted(missing_cols)}"
        )

    ranked = stability_df.sort_values(
        by=["frequency", "mean_rank", "protein_idx"],
        ascending=[False, True, True],
        kind="mergesort",
    )

    threshold_ids = ranked.loc[ranked["frequency"] >= pi, "protein_id"].astype(str).tolist()
    if len(threshold_ids) >= min_size:
        return threshold_ids, "threshold"

    fallback_n = min(int(topk_fallback), int(len(ranked)))
    fallback_ids = ranked.head(fallback_n)["protein_id"].astype(str).tolist()
    return fallback_ids, "topk_fallback"