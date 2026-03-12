"""
ards_simulation.py
==================
CLI script for the ARDS proteomics simulator.

Classes ``SimulationConfig`` and ``ARDSProteinSimulator`` live in
``src.core.simulation_utils`` and are imported here for convenience.

Example
-------
  python scripts/ards_simulation.py
  python scripts/ards_simulation.py --n_samples 300 --seed 7 --prefix run2
  python scripts/ards_simulation.py --frac_mean_shift 0.5 --frac_shape_change 0.3 --frac_interaction 0.2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.simulation_utils import ARDSProteinSimulator, SimulationConfig


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Gaussian-copula ARDS proteomics simulator with ground truth."
    )

    # Dataset
    ap.add_argument("--n_samples", type=int, default=600)
    ap.add_argument("--frac_ards", type=float, default=0.20)

    # Proteins / pathways
    ap.add_argument("--n_proteins", type=int, default=1000)
    ap.add_argument("--n_pathways", type=int, default=20)
    ap.add_argument("--pathway_corr_lo", type=float, default=0.30,
                    help="Lower bound for within-pathway correlation")
    ap.add_argument("--pathway_corr_hi", type=float, default=0.70,
                    help="Upper bound for within-pathway correlation")

    # ARDS prevalence
    ap.add_argument("--frac_ards_dependent", type=float, default=0.05,
                    help="Fraction of proteins with any ARDS effect")
    ap.add_argument("--frac_pathways_without_effects", type=float, default=0.50,
                    help="Target fraction of pathways with no ARDS-affected proteins")

    # Effect type breakdown (must sum to 1.0)
    ap.add_argument("--frac_mean_shift", type=float, default=0.50,
                    help="Fraction of affected proteins with mean-shift effect")
    ap.add_argument("--frac_shape_change", type=float, default=0.20,
                    help="Fraction of affected proteins with shape-change effect")
    ap.add_argument("--frac_interaction", type=float, default=0.30,
                    help="Fraction of affected proteins with interaction effect")

    # Effect-specific ranges
    ap.add_argument("--mean_shift_lo", type=float, default=0.2)
    ap.add_argument("--mean_shift_hi", type=float, default=1.0)
    ap.add_argument("--std_shift_lo", type=float, default=0.8)
    ap.add_argument("--std_shift_hi", type=float, default=1.5)
    ap.add_argument("--mixture_delta_lo", type=float, default=0.2)
    ap.add_argument("--mixture_delta_hi", type=float, default=1.0)
    ap.add_argument("--mixture_std_lo", type=float, default=0.4)
    ap.add_argument("--mixture_std_hi", type=float, default=0.8)
    ap.add_argument("--interaction_rho_lo", type=float, default=0.20)
    ap.add_argument("--interaction_rho_hi", type=float, default=0.80)

    # Misc
    ap.add_argument("--no_cross_pathway", action="store_true",
                    help="Disable preferring cross-pathway interactions")
    ap.add_argument("--seed", type=int, default=42)

    # Output
    ap.add_argument("--outdir", type=str, default=None,
                    help="Output directory (default: data/simulated)")
    ap.add_argument("--prefix", type=str, default="ards_sim",
                    help="Filename prefix for outputs")

    return ap.parse_args()


def _make_serializable(obj):
    """Convert numpy types to Python natives for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj


def save_outputs(
    outdir: Path,
    prefix: str,
    sim: ARDSProteinSimulator,
    df,
    gt: dict,
) -> None:
    """Save simulation data, ground-truth JSON, and ground-truth CSV."""
    outdir.mkdir(parents=True, exist_ok=True)

    # Data CSV (proteins + ards label)
    data_path = outdir / f"{prefix}.csv"
    df.to_csv(data_path, index=False)

    # Ground-truth CSV (one row per protein, human-readable)
    gt_df_path = outdir / f"{prefix}_ground_truth.csv"
    sim.ground_truth_df().to_csv(gt_df_path, index=False)

    # Ground-truth JSON (full detail, excluding large copula matrices)
    gt_json = {k: v for k, v in gt.items()
               if k not in ("Sigma_control", "Sigma_ards")}
    gt_json_path = outdir / f"{prefix}_truth.json"
    with open(gt_json_path, "w", encoding="utf-8") as f:
        json.dump(_make_serializable(gt_json), f, indent=2)

    # Copula matrices (numpy compressed)
    sigma_path = outdir / f"{prefix}_sigmas.npz"
    np.savez_compressed(
        sigma_path,
        Sigma_control=gt["Sigma_control"],
        Sigma_ards=gt["Sigma_ards"],
    )

    print(f"Wrote:\n"
          f"  {data_path}\n"
          f"  {gt_df_path}\n"
          f"  {gt_json_path}\n"
          f"  {sigma_path}")


def main() -> None:
    args = _parse_args()

    project_root = Path(__file__).resolve().parent.parent
    outdir = Path(args.outdir) if args.outdir else project_root / "data" / "simulated"

    cfg = SimulationConfig(
        n_samples=args.n_samples,
        frac_ards=args.frac_ards,
        n_proteins=args.n_proteins,
        n_pathways=args.n_pathways,
        pathway_corr_range=(args.pathway_corr_lo, args.pathway_corr_hi),
        frac_ards_dependent=args.frac_ards_dependent,
        frac_pathways_without_effects=args.frac_pathways_without_effects,
        effect_fractions={
            "mean_shift":   args.frac_mean_shift,
            "shape_change": args.frac_shape_change,
            "interaction":  args.frac_interaction,
        },
        mean_shift_range=(args.mean_shift_lo, args.mean_shift_hi),
        std_shift_range=(args.std_shift_lo, args.std_shift_hi),
        mixture_delta_range=(args.mixture_delta_lo, args.mixture_delta_hi),
        mixture_std_range=(args.mixture_std_lo, args.mixture_std_hi),
        interaction_rho_range=(args.interaction_rho_lo, args.interaction_rho_hi),
        prefer_cross_pathway_interactions=not args.no_cross_pathway,
        random_seed=args.seed,
    )

    sim = ARDSProteinSimulator(cfg)
    df, gt = sim.simulate()

    print(sim.summary())
    save_outputs(outdir, args.prefix, sim, df, gt)


if __name__ == "__main__":
    main()