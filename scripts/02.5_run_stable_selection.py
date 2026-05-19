"""Compute stability-based native selection and update full_data_ranking.parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute stability-based native selection from per-split selections "
            "and update full_data_ranking.parquet with significant=True for "
            "stably selected proteins."
        )
    )
    parser.add_argument("--method",     type=str,   required=True)
    parser.add_argument("--k",          type=int,   default=100)
    parser.add_argument("--tau",        type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path,  default=Path("results/selection"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.0 < args.tau < 1.0:
        raise ValueError(f"--tau must be in (0, 1), got {args.tau}.")
    if args.k <= 0:
        raise ValueError(f"--k must be > 0, got {args.k}.")

    method_dir = args.output_dir / args.method

    # ------------------------------------------------------------------ #
    # 1. Validate inputs                                                   #
    # ------------------------------------------------------------------ #
    for path in [
        method_dir / "selections.parquet",
        method_dir / "full_data_ranking.parquet",
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                f"Run 02_run_selection.py --method {args.method} first."
            )

    # ------------------------------------------------------------------ #
    # 2. Load data                                                         #
    # ------------------------------------------------------------------ #
    selections = pd.read_parquet(method_dir / "selections.parquet")
    full_data  = pd.read_parquet(method_dir / "full_data_ranking.parquet")

    # Verify rank is 0-indexed
    rank_min = selections["rank"].min()
    rank_max = selections["rank"].max()
    if rank_min != 0:
        raise ValueError(
            f"Expected rank to be 0-indexed (min=0), got min={rank_min}. "
            f"Check selections.parquet."
        )
    print(f"rank range: {rank_min} – {rank_max} (0-indexed confirmed)")

    n_splits   = selections["split_id"].nunique()
    n_proteins = len(full_data)

    print(f"n_splits   : {n_splits}")
    print(f"n_proteins : {n_proteins}")

    # ------------------------------------------------------------------ #
    # 3. Compute selection frequency                                       #
    #    frequency = fraction of splits where protein is in top-k         #
    # ------------------------------------------------------------------ #
    freq = (
        selections[selections["rank"] < args.k]
        .groupby("protein_idx")["split_id"]
        .nunique()
        .div(n_splits)
        .rename("frequency")
        .reset_index()
    )

    # ------------------------------------------------------------------ #
    # 4. Update full_data_ranking                                          #
    # ------------------------------------------------------------------ #
    # Drop frequency column if it already exists from a previous run
    if "frequency" in full_data.columns:
        full_data = full_data.drop(columns=["frequency"])

    full_data = full_data.merge(freq, on="protein_idx", how="left")
    full_data["frequency"]   = full_data["frequency"].fillna(0.0)
    full_data["significant"] = full_data["frequency"] >= args.tau

    full_data.to_parquet(method_dir / "full_data_ranking.parquet", index=False)

    # ------------------------------------------------------------------ #
    # 5. Summary and metadata                                              #
    # ------------------------------------------------------------------ #
    n_stable  = int(full_data["significant"].sum())
    fdr_bound = (args.k ** 2) / ((2 * args.tau - 1) * n_proteins)

    print(f"\nMethod : {args.method}")
    print(f"k      : {args.k}")
    print(f"tau    : {args.tau}")
    print(f"Stable proteins : {n_stable} / {n_proteins}")
    print(f"FDR bound       : {fdr_bound:.4f}")
    print(f"\nFrequency distribution:")
    for threshold in [0.3, 0.5, 0.7, 1.0]:
        n = int((full_data["frequency"] >= threshold).sum())
        print(f"  >= {threshold:.1f} : {n}")

    meta = {
        "method"   : args.method,
        "k"        : args.k,
        "tau"      : args.tau,
        "n_splits" : n_splits,
        "n_stable" : n_stable,
        "fdr_bound": round(fdr_bound, 6),
        "frequency_stats": {
            "n_ge_0_3": int((full_data["frequency"] >= 0.3).sum()),
            "n_ge_0_5": int((full_data["frequency"] >= 0.5).sum()),
            "n_ge_0_7": int((full_data["frequency"] >= 0.7).sum()),
            "n_ge_1_0": int((full_data["frequency"] >= 1.0).sum()),
        },
    }
    with (method_dir / "stable_selection_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nUpdated : {method_dir / 'full_data_ranking.parquet'}")
    print(f"Meta    : {method_dir / 'stable_selection_meta.json'}")


if __name__ == "__main__":
    main()