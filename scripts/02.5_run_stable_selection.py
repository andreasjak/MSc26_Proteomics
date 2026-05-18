"""Compute stability-based native selection and update full_data_ranking.parquet."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--k",      type=int, default=100)
    parser.add_argument("--tau",    type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("results/selection"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    method_dir = args.output_dir / args.method

    selections = pd.read_parquet(method_dir / "selections.parquet")
    full_data  = pd.read_parquet(method_dir / "full_data_ranking.parquet")

    n_splits = selections["split_id"].nunique()

    # Frequency = fraction of splits where protein is in top-k
    freq = (
        selections[selections["rank"] < args.k]
        .groupby("protein_idx")["split_id"]
        .nunique()
        .div(n_splits)
        .rename("frequency")
        .reset_index()
    )

    full_data = full_data.merge(freq, on="protein_idx", how="left")
    full_data["frequency"]  = full_data["frequency"].fillna(0.0)
    full_data["significant"] = full_data["frequency"] >= args.tau

    full_data.to_parquet(method_dir / "full_data_ranking.parquet", index=False)

    n_stable    = int(full_data["significant"].sum())
    fdr_bound   = (args.k ** 2) / ((2 * args.tau - 1) * len(full_data))

    print(f"Method: {args.method} | k={args.k} | tau={args.tau}")
    print(f"Stable proteins : {n_stable} / {len(full_data)}")
    print(f"FDR bound       : {fdr_bound:.4f}")

    meta = {
        "method": args.method, "k": args.k, "tau": args.tau,
        "n_splits": n_splits, "n_stable": n_stable, "fdr_bound": fdr_bound,
    }
    with (method_dir / "stable_selection_meta.json").open("w") as f:
        import json; json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()