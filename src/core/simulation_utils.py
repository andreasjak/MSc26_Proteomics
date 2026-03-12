"""
simulation_utils
================
Gaussian-copula simulator for high-dimensional proteomics with ARDS effects
and pathway-correlated protein structure.

Protein effect types (ARDS vs control)
---------------------------------------
  null         – identical distribution in both groups
  mean_shift   – shifted mean (and optionally variance) in ARDS
  shape_change – bimodal (Gaussian mixture) marginal in ARDS, same mean as
                 control  (E[X] = 0 preserved; Var[X] increases)
  interaction  – altered pairwise correlation in ARDS (ρ_ctrl → ρ_ards)
                 preferentially across pathways (cross-talk)

Architecture
------------
  1. Proteins are assigned to pathways. Within each pathway a compound-
     symmetry (exchangeable) correlation structure encodes co-regulation.
  2. A Gaussian copula encodes all dependencies: sample Z ~ MVN(0, Σ),
     then apply per-protein marginal transforms to obtain X.
  3. Σ_control and Σ_ards share the same block-diagonal pathway structure.
     Interaction pairs modify off-diagonal entries in Σ_ards only.
  4. Mean-shift  : X = μ + σ·Z  (linear; preserves Pearson correlation)
  5. Shape-change: X = F_mix⁻¹(Φ(Z))  (quantile transform; preserves
                 Spearman / rank correlation, changes Pearson)

Ground truth
------------
  simulate() returns (df, ground_truth).
  ground_truth contains everything needed for sanity checks:
    - protein_to_pathway / pathway_to_proteins
    - pathway_corr            : within-pathway ρ per pathway
    - null/mean_shift/shape_change/interaction protein name lists
    - mean_shift_params        : {protein: {mean_ards, std_ards, ...}}
    - shape_change_params      : {protein: {weights, means, stds}}
    - interaction_pairs        : list of dicts with p1, p2, ρ_ctrl, ρ_ards
    - Sigma_control, Sigma_ards: full copula matrices  (n_proteins × n_proteins)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationConfig:
    """All parameters needed to reproduce a simulation run."""

    # ── Dataset ───────────────────────────────────────────────────────────────
    n_samples: int   = 600
    frac_ards: float = 0.20

    # ── Proteins / pathways ───────────────────────────────────────────────────
    n_proteins:         int                  = 1000
    n_pathways:         int                  = 20
    pathway_corr_range: Tuple[float, float]  = (0.30, 0.70)  # within-pathway ρ

    # ── ARDS prevalence ───────────────────────────────────────────────────────
    frac_ards_dependent: float = 0.10  # fraction of proteins with any ARDS effect
    frac_pathways_without_effects: float = 0.50  # target fraction of pathways with no ARDS-affected proteins

    # ── Effect type breakdown (must sum to 1.0) ───────────────────────────────
    # Interaction proteins are paired; an odd surplus is reallocated to mean_shift.
    effect_fractions: Dict[str, float] = field(default_factory=lambda: {
        "mean_shift":   0.40,
        "shape_change": 0.30,
        "interaction":  0.30,
    })

    # ── Effect-specific parameters ────────────────────────────────────────────
    mean_shift_range:      Tuple[float, float] = (0.8, 2.0)  # |Δμ| in ARDS
    std_shift_range:       Tuple[float, float] = (0.8, 1.5)  # σ in ARDS
    mixture_delta_range:   Tuple[float, float] = (0.8, 2.0)  # half-spacing of modes
    mixture_std_range:     Tuple[float, float] = (0.4, 0.8)  # per-component σ
    interaction_rho_range: Tuple[float, float] = (0.40, 0.80)  # |ρ_ards|, sign random

    # ── Misc ──────────────────────────────────────────────────────────────────
    prefer_cross_pathway_interactions: bool = True  # cross-pathway pairs first
    random_seed: int = 42

    def __post_init__(self):
        s = sum(self.effect_fractions.values())
        if not np.isclose(s, 1.0, atol=1e-6):
            raise ValueError(f"effect_fractions must sum to 1.0, got {s:.6f}.")
        if not (0.0 <= self.frac_pathways_without_effects < 1.0):
            raise ValueError(
                "frac_pathways_without_effects must be in [0.0, 1.0)."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Simulator
# ─────────────────────────────────────────────────────────────────────────────

class ARDSProteinSimulator:
    """
    Simulate high-dimensional proteomics with controlled ARDS-specific effects.

    Parameters
    ----------
    cfg : SimulationConfig

    Main API
    --------
    df, gt = sim.simulate()    # returns DataFrame + ground-truth dict
    print(sim.summary())       # human-readable breakdown
    sim.ground_truth_df()      # flat DataFrame, one row per protein
    """

    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_seed)
        self.ground_truth: Dict[str, Any] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def simulate(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run simulation.

        Returns
        -------
        df : pd.DataFrame  (n_samples × [n_proteins + 1])
            Columns seq.0001 … seq.{n_proteins:04d}, 'ards' (0/1).
        ground_truth : dict
            Full specification of protein effects and simulation parameters.
        """
        cfg    = self.cfg
        P      = cfg.n_proteins
        n_ards = int(cfg.n_samples * cfg.frac_ards)
        n_ctrl = cfg.n_samples - n_ards

        # 1. Pathway structure
        prot2path, path2prots = self._assign_pathways(P, cfg.n_pathways)
        path_corr = self._sample_pathway_corrs(path2prots)

        # 2. Assign ARDS effects to proteins
        effects = self._assign_effects(P, prot2path, path2prots, path_corr)

        # 3. Build copula correlation matrices
        Sigma_ctrl, Sigma_ards = self._build_sigmas(
            P, path2prots, path_corr, effects
        )

        # 4. Sample latent Gaussians from copula (Cholesky is faster and more
        #    numerically stable than the default SVD decomposition)
        Z_ctrl = self.rng.multivariate_normal(
            np.zeros(P), Sigma_ctrl, n_ctrl, method="cholesky"
        )
        Z_ards = self.rng.multivariate_normal(
            np.zeros(P), Sigma_ards, n_ards, method="cholesky"
        )

        # 5. Apply per-protein marginal transforms
        X_ctrl = self._apply_marginals(Z_ctrl, effects, group="control")
        X_ards = self._apply_marginals(Z_ards, effects, group="ards")

        # 6. Assemble, shuffle, return
        X    = np.vstack([X_ctrl, X_ards])
        y    = np.array([0] * n_ctrl + [1] * n_ards)
        cols = [f"seq.{i+1:04d}" for i in range(P)]
        df   = pd.DataFrame(X, columns=cols)
        df["ards"] = y
        df = df.sample(frac=1, random_state=cfg.random_seed).reset_index(drop=True)

        self.ground_truth = self._build_ground_truth(
            cols, prot2path, path2prots, path_corr,
            effects, Sigma_ctrl, Sigma_ards, n_ctrl, n_ards,
        )
        return df, self.ground_truth

    # ── Reporting ─────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable summary of the ground truth."""
        gt  = self.ground_truth
        cfg = self.cfg
        P   = cfg.n_proteins
        n_int   = len(gt["interaction_proteins"])
        n_pairs = len(gt["interaction_pairs"])

        lines = [
            "=" * 54,
            "  ARDS Protein Simulation – Ground Truth",
            "=" * 54,
            f"  Samples  : {cfg.n_samples:6d}"
            f"  (ARDS={gt['n_ards']}, Control={gt['n_control']})",
            f"  Proteins : {P:6d}",
            f"  Pathways : {cfg.n_pathways:6d}",
            "",
            "  Effect breakdown",
            f"    Null         : {len(gt['null_proteins']):5d}"
            f"  ({100*len(gt['null_proteins'])/P:.1f}%)",
            f"    Mean-shift   : {len(gt['mean_shift_proteins']):5d}"
            f"  ({100*len(gt['mean_shift_proteins'])/P:.1f}%)",
            f"    Shape-change : {len(gt['shape_change_proteins']):5d}"
            f"  ({100*len(gt['shape_change_proteins'])/P:.1f}%)",
            f"    Interaction  : {n_int:5d}"
            f"  ({100*n_int/P:.1f}%)  →  {n_pairs} pairs",
            "=" * 54,
        ]
        return "\n".join(lines)

    def ground_truth_df(self) -> pd.DataFrame:
        """
        Flat DataFrame with one row per protein.

        Columns: protein, pathway, effect_type, [effect-specific params]

        Convenient for sanity-checking downstream analysis results against
        the known ground truth.
        """
        gt   = self.ground_truth
        rows = []

        def row(name, etype, extra=None):
            return {
                "protein":     name,
                "pathway":     gt["protein_to_pathway"][name],
                "effect_type": etype,
                **(extra or {}),
            }

        for n in gt["null_proteins"]:
            rows.append(row(n, "null"))

        for n in gt["mean_shift_proteins"]:
            p = gt["mean_shift_params"][n]
            rows.append(row(n, "mean_shift", {
                "mean_control": p["mean_control"],
                "std_control":  p["std_control"],
                "mean_ards":    round(p["mean_ards"], 4),
                "std_ards":     round(p["std_ards"],  4),
            }))

        for n in gt["shape_change_proteins"]:
            p = gt["shape_change_params"][n]
            rows.append(row(n, "shape_change", {
                "mixture_means": str(p["means"]),
                "mixture_stds":  str(p["stds"]),
            }))

        for pair in gt["interaction_pairs"]:
            for prot, partner in [
                (pair["protein_1"], pair["protein_2"]),
                (pair["protein_2"], pair["protein_1"]),
            ]:
                rows.append(row(prot, "interaction", {
                    "partner":       partner,
                    "rho_control":   round(pair["rho_control"], 4),
                    "rho_ards":      round(pair["rho_ards"],    4),
                    "cross_pathway": pair["cross_pathway"],
                }))

        return (
            pd.DataFrame(rows)
            .sort_values("protein")
            .reset_index(drop=True)
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: pathway assignment
    # ─────────────────────────────────────────────────────────────────────────

    def _assign_pathways(
        self, P: int, K: int
    ) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
        """
        Randomly assign P proteins to K pathways (roughly equal sizes).
        Proteins are first permuted, then split into K consecutive chunks.
        """
        idx        = self.rng.permutation(P)
        splits     = np.array_split(idx, K)
        prot2path  = {}
        path2prots = {}
        for k, members in enumerate(splits):
            members = list(map(int, members))
            path2prots[k] = members
            for p in members:
                prot2path[p] = k
        return prot2path, path2prots

    def _sample_pathway_corrs(
        self, path2prots: Dict[int, List[int]]
    ) -> Dict[int, float]:
        """Sample a within-pathway ρ for each pathway from the configured range."""
        lo, hi = self.cfg.pathway_corr_range
        return {k: float(self.rng.uniform(lo, hi)) for k in path2prots}

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: effect assignment
    # ─────────────────────────────────────────────────────────────────────────

    def _assign_effects(
        self,
        P:          int,
        prot2path:  Dict[int, int],
        path2prots: Dict[int, List[int]],
        path_corr:  Dict[int, float],
    ) -> Dict[str, Any]:
        cfg = self.cfg

        # Restrict ARDS effects to a subset of pathways so some pathways remain
        # entirely unaffected.
        n_without = min(
            int(round(cfg.n_pathways * cfg.frac_pathways_without_effects)),
            cfg.n_pathways - 1,
        )
        permuted = self.rng.permutation(cfg.n_pathways)
        pathways_without_effects = set(permuted[:n_without].tolist())
        pathways_with_effects = permuted[n_without:].tolist()
        candidate_proteins = [
            p for k in pathways_with_effects for p in path2prots[k]
        ]

        if not candidate_proteins:
            raise ValueError(
                "No candidate proteins available for ARDS effects. "
                "Lower frac_pathways_without_effects or increase n_pathways."
            )

        # Total ARDS-affected proteins
        n_affected = max(2, round(P * cfg.frac_ards_dependent))
        n_affected = min(n_affected, len(candidate_proteins))
        affected   = list(map(
            int,
            self.rng.choice(candidate_proteins, size=n_affected, replace=False)
        ))
        self.rng.shuffle(affected)

        # Split by effect type
        ef    = cfg.effect_fractions
        n_ms  = round(n_affected * ef["mean_shift"])
        n_sc  = round(n_affected * ef["shape_change"])
        n_int = n_affected - n_ms - n_sc

        # Interactions require pairs → must be even; reallocate surplus to mean_shift
        if n_int % 2 != 0:
            n_int -= 1
            n_ms  += 1

        ms_prots  = affected[:n_ms]
        sc_prots  = affected[n_ms : n_ms + n_sc]
        int_prots = affected[n_ms + n_sc : n_ms + n_sc + n_int]

        ms_params = self._sample_mean_shift_params(ms_prots)
        sc_params = self._sample_shape_change_params(sc_prots)
        int_pairs = self._pair_interaction_proteins(int_prots, prot2path, path_corr)

        affected_set = set(ms_prots) | set(sc_prots) | set(int_prots)
        null_prots   = [p for p in range(P) if p not in affected_set]

        return {
            "null":                 null_prots,
            "mean_shift":           ms_prots,
            "mean_shift_params":    ms_params,
            "shape_change":         sc_prots,
            "shape_change_params":  sc_params,
            "interaction_proteins": int_prots,
            "interaction_pairs":    int_pairs,  # List[(p1, p2, ρ_ctrl, ρ_ards)]
            "pathways_with_effects": sorted(pathways_with_effects),
            "pathways_without_effects": sorted(pathways_without_effects),
        }

    def _sample_mean_shift_params(self, prots: List[int]) -> Dict[int, dict]:
        params = {}
        for p in prots:
            sign = int(self.rng.choice([-1, 1]))
            params[p] = {
                "mean_control": 0.0,
                "std_control":  1.0,
                "mean_ards": sign * float(self.rng.uniform(*self.cfg.mean_shift_range)),
                "std_ards":  float(self.rng.uniform(*self.cfg.std_shift_range)),
            }
        return params

    def _sample_shape_change_params(self, prots: List[int]) -> Dict[int, dict]:
        """
        Bimodal mixture with means at ±delta.
        Symmetric weights → E[X] = 0 matches control mean.
        Wider modes → Var[X] > 1 (detectable by variance tests, not t-test).
        """
        params = {}
        for p in prots:
            delta = float(self.rng.uniform(*self.cfg.mixture_delta_range))
            s     = float(self.rng.uniform(*self.cfg.mixture_std_range))
            params[p] = {
                "weights": [0.5, 0.5],
                "means":   [-delta, delta],
                "stds":    [s, s],
                "note":    "bimodal; E[X]=0 matches control; Var[X] > 1",
            }
        return params

    def _pair_interaction_proteins(
        self,
        int_prots: List[int],
        prot2path: Dict[int, int],
        path_corr: Dict[int, float],
    ) -> List[Tuple[int, int, float, float]]:
        """
        Pair interaction proteins.

        Strategy:
          1. If prefer_cross_pathway_interactions: greedy scan for cross-pathway pairs
             (ρ_ctrl = 0; ARDS creates new cross-pathway dependency).
          2. Remaining unpaired proteins: within-pathway pairs
             (ρ_ctrl = pathway ρ; ARDS alters the existing within-pathway correlation).
        """
        pairs  : List[Tuple[int, int, float, float]] = []
        paired : set = set()
        prots  = list(int_prots)

        if self.cfg.prefer_cross_pathway_interactions:
            for i, p1 in enumerate(prots):
                if p1 in paired:
                    continue
                for p2 in prots[i + 1:]:
                    if p2 in paired:
                        continue
                    if prot2path[p1] != prot2path[p2]:
                        rho_ards = self._sample_rho_ards()
                        rho_ctrl = - rho_ards * 0.5
                        pairs.append((p1, p2, rho_ctrl, rho_ards))
                        paired.update([p1, p2])
                        break   # move to next p1

        # Fallback: pair remaining proteins (may be within-pathway)
        unpaired = [p for p in prots if p not in paired]
        for i in range(0, len(unpaired) - 1, 2):
            p1, p2   = unpaired[i], unpaired[i + 1]
            same_k   = prot2path[p1] == prot2path[p2]
            rho_ards = self._sample_rho_ards()
            #rho_ctrl = path_corr[prot2path[p1]] if same_k else 0.0
            rho_ctrl = - rho_ards * 0.5
            pairs.append((p1, p2, rho_ctrl, rho_ards))

        return pairs

    def _sample_rho_ards(self) -> float:
        """Random signed correlation for ARDS interaction."""
        sign = int(self.rng.choice([-1, 1]))
        return sign * float(self.rng.uniform(*self.cfg.interaction_rho_range))

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: correlation matrices
    # ─────────────────────────────────────────────────────────────────────────

    def _build_sigmas(
        self,
        P:          int,
        path2prots: Dict[int, List[int]],
        path_corr:  Dict[int, float],
        effects:    Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build Σ_control and Σ_ards.

        Base: block-diagonal compound symmetry from pathways.
        ARDS: same base, with interaction pair correlations overwritten.
        Both matrices are projected to the nearest correlation matrix (PD + unit diagonal).
        """
        def base_sigma() -> np.ndarray:
            S = np.eye(P)
            for k, members in path2prots.items():
                rho = path_corr[k]
                for i in members:
                    for j in members:
                        if i != j:
                            S[i, j] = rho
            return S

        Sigma_ctrl = base_sigma()
        Sigma_ards = base_sigma()

        for p1, p2, rho_ctrl, rho_ards in effects["interaction_pairs"]:
            # rho_ctrl may override base for cross-pathway (sets to 0 explicitly)
            Sigma_ctrl[p1, p2] = Sigma_ctrl[p2, p1] = rho_ctrl
            Sigma_ards[p1, p2] = Sigma_ards[p2, p1] = rho_ards

        return self._nearest_corr(Sigma_ctrl), self._nearest_corr(Sigma_ards)

    @staticmethod
    def _nearest_corr(A: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Project A onto the nearest positive-definite correlation matrix.
        Steps: symmetrize → clip negative eigenvalues → rescale to unit diagonal.
        """
        A = (A + A.T) / 2
        vals, vecs = np.linalg.eigh(A)
        vals = np.maximum(vals, eps)
        A = vecs @ np.diag(vals) @ vecs.T
        # Rescale to unit diagonal (correlation matrix)
        d = np.sqrt(np.diag(A))
        A = A / np.outer(d, d)
        return A

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: marginal transforms
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_marginals(
        self, Z: np.ndarray, effects: Dict[str, Any], group: str
    ) -> np.ndarray:
        """
        Apply per-protein marginal transforms on top of the copula samples Z.

        Control: identity (X = Z; standard normal).
        ARDS:
          mean_shift   → X = μ + σ·Z  (linear; Pearson correlation preserved)
          shape_change → X = F_mix⁻¹(Φ(Z))  (Spearman preserved, Pearson changes)
          interaction  → identity (effect is in Σ_ards, not the marginal)
          null         → identity
        """
        X = Z.copy()

        if group == "ards":
            for p, par in effects["mean_shift_params"].items():
                X[:, p] = par["mean_ards"] + par["std_ards"] * Z[:, p]

            for p, par in effects["shape_change_params"].items():
                u       = norm.cdf(Z[:, p])
                X[:, p] = self._mixture_ppf(
                    u, par["weights"], par["means"], par["stds"]
                )

        return X

    @staticmethod
    def _mixture_ppf(
        u:       np.ndarray,
        weights: List[float],
        means:   List[float],
        stds:    List[float],
        n_grid:  int = 3_000,
    ) -> np.ndarray:
        """
        Quantile function of a Gaussian mixture via fast grid interpolation.

        Complexity: O(K·n_grid) to build the CDF grid, O(n·log n_grid) to look up.
        Default n_grid=3000 gives ~10⁻³ absolute accuracy for the transforms used.
        """
        bound    = float(max(abs(m) for m in means) + 5 * max(stds))
        x_grid   = np.linspace(-bound, bound, n_grid)
        cdf_grid = sum(
            w * norm.cdf(x_grid, m, s)
            for w, m, s in zip(weights, means, stds)
        )
        # Clip to avoid interpolation boundary issues
        u_safe = np.clip(u, cdf_grid[0] + 1e-10, cdf_grid[-1] - 1e-10)
        return np.interp(u_safe, cdf_grid, x_grid)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: ground truth packaging
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ground_truth(
        self,
        cols:       List[str],
        prot2path:  Dict[int, int],
        path2prots: Dict[int, List[int]],
        path_corr:  Dict[int, float],
        effects:    Dict[str, Any],
        Sigma_ctrl: np.ndarray,
        Sigma_ards: np.ndarray,
        n_ctrl:     int,
        n_ards:     int,
    ) -> Dict[str, Any]:
        return {
            # ── Protein / pathway identifiers ──────────────────────────────────
            "protein_names":       cols,
            "protein_to_pathway":  {cols[p]: k for p, k in prot2path.items()},
            "pathway_to_proteins": {
                k: [cols[p] for p in ps] for k, ps in path2prots.items()
            },
            "pathway_corr": path_corr,   # {pathway_id: ρ_within}
            "pathways_with_effects": effects["pathways_with_effects"],
            "pathways_without_effects": effects["pathways_without_effects"],

            # ── Effect protein lists (by name) ─────────────────────────────────
            "null_proteins":         [cols[p] for p in effects["null"]],
            "mean_shift_proteins":   [cols[p] for p in effects["mean_shift"]],
            "shape_change_proteins": [cols[p] for p in effects["shape_change"]],
            "interaction_proteins":  [cols[p] for p in effects["interaction_proteins"]],

            # ── Per-protein parameters ─────────────────────────────────────────
            "mean_shift_params": {
                cols[p]: par for p, par in effects["mean_shift_params"].items()
            },
            "shape_change_params": {
                cols[p]: par for p, par in effects["shape_change_params"].items()
            },
            "interaction_pairs": [
                {
                    "protein_1":    cols[p1],
                    "protein_2":    cols[p2],
                    "rho_control":  rho_ctrl,
                    "rho_ards":     rho_ards,
                    "cross_pathway": prot2path[p1] != prot2path[p2],
                }
                for p1, p2, rho_ctrl, rho_ards in effects["interaction_pairs"]
            ],

            # ── Full copula matrices ───────────────────────────────────────────
            "Sigma_control": Sigma_ctrl,
            "Sigma_ards":    Sigma_ards,

            # ── Sample counts ──────────────────────────────────────────────────
            "n_control": n_ctrl,
            "n_ards":    n_ards,
        }
