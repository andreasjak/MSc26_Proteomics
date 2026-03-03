"""
Univariate Mutual Information pipeline for proteomics data.

Operates on seen.csv (train + validation set produced by preprocess.py).
Performs:
  1. Data loading
  2. Univariate MI estimation (Kraskov k-NN) for every protein vs. ARDS label
  3. Permutation-based significance testing
  4. Multiple testing correction (FDR or Bonferroni)

Outputs (when --save-results):
  results/<results-subdir>/results_mi_uni_test.csv
  results/<results-subdir>/selected_features_k{K}.csv
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import logging
import time

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.multitest import multipletests

from src.core.data_utils import get_protein_features, load_data
from src.core.logging_utils import setup_logging


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def compute_mi(
    data: pd.DataFrame,
    n_neighbors: int,
    random_state: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Compute univariate MI between each protein feature and the ARDS label.

    Returns
    -------
    mi_results : pd.DataFrame
        Columns: Protein, MI — sorted descending by MI.
    X_values : np.ndarray
        Cleaned protein feature matrix (samples × proteins), column order
        matching mi_results["Protein"].
    y_values : np.ndarray
        Integer ARDS labels.
    """
    protein_cols = get_protein_features(data)
    logger.info("Protein columns to score: %d", len(protein_cols))

    X = data[protein_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    y = data["ards"].astype(int).values

    logger.info("Computing univariate MI (k=%d) …", n_neighbors)
    mi_scores = mutual_info_classif(
        X.values,
        y,
        discrete_features=False,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )

    mi_results = (
        pd.DataFrame({"Protein": protein_cols, "MI": mi_scores})
        .sort_values("MI", ascending=False)
        .reset_index(drop=True)
    )

    logger.info("Top 15 proteins by MI:\n%s", mi_results.head(15).to_string(index=False))

    return mi_results, X.values, y

def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    mi_results: pd.DataFrame,
    n_perm: int,
    n_neighbors: int,
    random_state: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Permutation test for MI significance.

    Shuffles the label vector ``n_perm`` times, re-computes MI each time,
    and derives empirical p-values as (count + 1) / (n_perm + 1).

    Returns a copy of *mi_results* with an added ``p_perm`` column.
    """
    rng = np.random.default_rng(random_state)
    mi_obs = mi_results["MI"].values
    prot_order = mi_results["Protein"].tolist()

    # Reorder X columns to match mi_results order.
    # X was built from protein_cols in compute_mi; here we need the same
    # column mapping.  Since mi_results is just a sort of those columns,
    # and X is a plain ndarray, we need to track indices.
    # Build a column-index lookup from the original protein_cols order.
    # protein_cols in mi_results may be re-sorted; X columns follow the
    # original get_protein_features order.  Safest: just reuse X as-is
    # and reorder mi_obs to match X column order.
    #
    # Actually, mi_obs is already aligned to mi_results["Protein"], but X
    # columns follow get_protein_features order.  Re-sort mi_obs to X order.
    # — Simpler approach: pass X already ordered to match mi_results.
    # The caller (main) can handle this.  For now we assume X columns and
    # mi_results["Protein"] are in corresponding order (both sorted desc MI).
    # We re-sort X columns at the call site.

    perm_counts = np.zeros(len(prot_order), dtype=int)

    logger.info("Running %d permutations …", n_perm)
    t0 = time.time()
    for b in range(n_perm):
        if n_perm >= 10 and (b + 1) % (n_perm // 10) == 0:
            logger.info(
                "  Permutation %d/%d (%d%%) — elapsed %.1f s",
                b + 1, n_perm, int((b + 1) / n_perm * 100), time.time() - t0,
            )
        y_perm = rng.permutation(y)
        mi_perm = mutual_info_classif(
            X,
            y_perm,
            discrete_features=False,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
        perm_counts += (mi_perm >= mi_obs)

    p_perm = (perm_counts + 1) / (n_perm + 1)

    mi_results = mi_results.copy()
    mi_results["p_perm"] = p_perm
    return mi_results


def adaptive_permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    mi_results: pd.DataFrame,
    min_perm: int,
    max_perm: int,
    alpha_stop: float,
    n_neighbors: int,
    random_state: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Adaptive permutation test for MI significance.

    Proteins are tested until their significance can be decided or
    *max_perm* permutations are reached.  After *min_perm* permutations
    the following bounds are checked for each still-active protein:

        p_min = hits / max_perm
        p_max = (hits + remaining) / max_perm

    If ``p_min > alpha_stop`` the protein is non-significant → stop.
    If ``p_max < alpha_stop`` the protein is significant → stop.

    Decided proteins are removed from the MI computation matrix to
    save compute on subsequent permutations.

    Final p-values are ``(hits + 1) / (n_done + 1)``.

    Returns a copy of *mi_results* with added ``p_perm`` and ``n_perm_done``
    columns.
    """
    rng = np.random.default_rng(random_state)
    n_proteins = X.shape[1]
    mi_obs = mi_results["MI"].values.copy()

    # Track per-protein state using original indices
    hits = np.zeros(n_proteins, dtype=int)
    n_done = np.zeros(n_proteins, dtype=int)
    active_mask = np.ones(n_proteins, dtype=bool)

    # Working copies that shrink as proteins are decided
    X_active = X.copy()
    mi_obs_active = mi_obs.copy()
    # Map from current active position → original protein index
    active_indices = np.arange(n_proteins)

    logger.info(
        "Running adaptive permutation test (min=%d, max=%d, alpha_stop=%.4f) "
        "on %d proteins …",
        min_perm, max_perm, alpha_stop, n_proteins,
    )
    t0 = time.time()
    log_interval = max(1, max_perm // 10)

    for b in range(max_perm):
        perm_idx = b + 1

        # Progress logging
        if perm_idx % log_interval == 0 or perm_idx == max_perm:
            logger.info(
                "  Permutation %d/%d (%d%%) — %d/%d proteins active — elapsed %.1f s",
                perm_idx, max_perm,
                int(perm_idx / max_perm * 100),
                int(active_mask.sum()), n_proteins,
                time.time() - t0,
            )

        y_perm = rng.permutation(y)
        mi_perm = mutual_info_classif(
            X_active,
            y_perm,
            discrete_features=False,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )

        # Update counts for currently active proteins
        cur_hits = (mi_perm >= mi_obs_active).astype(int)
        hits[active_indices] += cur_hits
        n_done[active_indices] += 1

        # Adaptive stopping check after min_perm
        if perm_idx >= min_perm:
            remaining = max_perm - n_done[active_indices]
            p_min = hits[active_indices] / max_perm
            p_max = (hits[active_indices] + remaining) / max_perm

            stop = (p_min > alpha_stop) | (p_max < alpha_stop)

            if stop.any():
                # Mark stopped proteins as inactive
                stopped_orig = active_indices[stop]
                active_mask[stopped_orig] = False

                # Shrink working arrays
                keep = ~stop
                X_active = X_active[:, keep]
                mi_obs_active = mi_obs_active[keep]
                active_indices = active_indices[keep]

            if len(active_indices) == 0:
                logger.info(
                    "  All proteins decided after %d permutations.", perm_idx,
                )
                break

    p_perm = (hits + 1) / (n_done + 1)

    mi_results = mi_results.copy()
    mi_results["p_perm"] = p_perm
    mi_results["n_perm_done"] = n_done.astype(int)

    n_early = int((n_done < max_perm).sum())
    logger.info(
        "Adaptive permutation test complete: %d/%d proteins stopped early. "
        "Permutations per protein — min=%d, median=%d, max=%d.",
        n_early, n_proteins,
        int(n_done.min()), int(np.median(n_done)), int(n_done.max()),
    )
    return mi_results


def correct_pvalues(
    results: pd.DataFrame,
    method: str,
    alpha: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Apply multiple testing correction to permutation p-values.

    Parameters
    ----------
    method : str
        ``'fdr'`` → Benjamini-Hochberg; ``'bonferroni'`` → Bonferroni.
    """
    method_map = {"fdr": "fdr_bh", "bonferroni": "bonferroni"}
    mt_method = method_map[method.lower()]

    results = results.copy()

    # Handle NaN p-values (untested proteins from --n-proteins filtering)
    tested = results["p_perm"].notna()
    results["ADJ_P"] = np.nan

    if tested.any():
        _, adj_p, _, _ = multipletests(
            results.loc[tested, "p_perm"].values, method=mt_method,
        )
        results.loc[tested, "ADJ_P"] = adj_p

    results = (
        results
        .sort_values(["ADJ_P", "MI"], ascending=[True, False])
        .reset_index(drop=True)
    )

    n_tested = int(tested.sum())
    n_sig = int((results["ADJ_P"] < alpha).sum())
    logger.info(
        "Correction: %s | alpha=%.3f | significant proteins: %d / %d tested (%d total)",
        mt_method, alpha, n_sig, n_tested, len(results),
    )
    logger.info(
        "Top 15 after correction:\n%s",
        results.head(15).to_string(index=False),
    )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start = time.time()

    parser = argparse.ArgumentParser(
        description="Univariate MI analysis with permutation significance testing."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/seen.csv"),
        help="Path to the seen (train+val) CSV (default: data/processed/seen.csv).",
    )
    parser.add_argument(
        "--results-subdir",
        type=str,
        default="mi_uni",
        help="Subdirectory under results/ for output files (default: mi).",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save outputs to disk and log to file; otherwise log to terminal only.",
    )
    parser.add_argument(
        "--log-subdir",
        type=str,
        default="mi_uni",
        help="Subdirectory under logs/ for log files (default: mi).",
    )
    parser.add_argument(
        "--correction-method",
        type=str,
        choices=["fdr", "bonferroni"],
        default="fdr",
        help="Multiple testing correction method (default: fdr).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for adjusted p-values (default: 0.05).",
    )
    parser.add_argument(
        "--n-proteins",
        type=int,
        default=None,
        help="Only run permutation test on the top N proteins by MI "
             "(default: None = test all proteins).",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=1000,
        help="Number of permutations (fixed mode, default: 1000). "
             "Ignored when --adaptive is set.",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Enable adaptive permutation testing with early stopping.",
    )
    parser.add_argument(
        "--min-perm",
        type=int,
        default=200,
        help="Minimum permutations before adaptive stopping (default: 200).",
    )
    parser.add_argument(
        "--max-perm",
        type=int,
        default=2000,
        help="Maximum permutations for adaptive mode (default: 2000).",
    )
    parser.add_argument(
        "--alpha-stop",
        type=float,
        default=None,
        help="Significance threshold for adaptive early stopping "
             "(default: same as --alpha).",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=3,
        help="k-NN neighbours for MI estimation / Kraskov estimator (default: 3).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for MI estimation and permutations (default: 42).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Number of top proteins to save as selected features (default: 10).",
    )
    args = parser.parse_args()

    logger = setup_logging(args.save_results, args.log_subdir, "mi_uni_tests")

    logger.info("Starting mi_uni_tests.py")
    logger.info(
        "Args: data_path=%s  correction=%s  alpha=%s  n_perm=%d  "
        "n_neighbors=%d  random_state=%d  k=%d  save_results=%s  "
        "n_proteins=%s  adaptive=%s  min_perm=%d  max_perm=%d  alpha_stop=%s",
        args.data_path, args.correction_method, args.alpha, args.n_perm,
        args.n_neighbors, args.random_state, args.k, args.save_results,
        args.n_proteins, args.adaptive, args.min_perm, args.max_perm,
        args.alpha_stop,
    )

    # Step 1: Load data
    data = load_data(args.data_path, logger)

    # Step 2: Compute univariate MI
    mi_results, X_prot, y = compute_mi(
        data, args.n_neighbors, args.random_state, logger,
    )

    # Reorder X columns to match mi_results sort order for permutation test.
    # X_prot columns follow get_protein_features order; mi_results is sorted
    # by MI descending.  Build index mapping.
    protein_cols = get_protein_features(data)
    col_idx = {name: i for i, name in enumerate(protein_cols)}
    reorder = [col_idx[p] for p in mi_results["Protein"]]
    X_perm = X_prot[:, reorder]

    # Optional: restrict permutation test to top-N proteins
    mi_results_full = mi_results  # keep full list (all proteins with MI)
    if args.n_proteins is not None:
        n_test = min(args.n_proteins, len(mi_results))
        logger.info(
            "Restricting permutation test to top %d / %d proteins by MI.",
            n_test, len(mi_results),
        )
        mi_results_test = mi_results.head(n_test).reset_index(drop=True)
        X_test = X_perm[:, :n_test]
    else:
        mi_results_test = mi_results
        X_test = X_perm

    # Step 3: Permutation test
    if args.adaptive:
        # Default alpha_stop to alpha / (nbr of proteins) if not set
        if args.alpha_stop is None:
            if args.n_proteins is not None:
                args.alpha_stop = args.alpha / args.n_proteins  
            else:
                args.alpha_stop = args.alpha / len(mi_results)

        mi_results_test = adaptive_permutation_test(
            X_test, y, mi_results_test,
            args.min_perm, args.max_perm, args.alpha_stop,
            args.n_neighbors, args.random_state, logger,
        )
    else:
        mi_results_test = permutation_test(
            X_test, y, mi_results_test,
            args.n_perm, args.n_neighbors, args.random_state, logger,
        )

    # Merge untested proteins back (they get NaN p-values)
    if args.n_proteins is not None:
        tested_proteins = set(mi_results_test["Protein"])
        untested = mi_results_full[
            ~mi_results_full["Protein"].isin(tested_proteins)
        ].copy()
        mi_results = pd.concat(
            [mi_results_test, untested], ignore_index=True,
        )
    else:
        mi_results = mi_results_test

    # Step 4: Multiple testing correction
    mi_results = correct_pvalues(
        mi_results, args.correction_method, args.alpha, logger,
    )

    # Step 5: Save results
    if args.save_results:
        results_dir = Path("results") / args.results_subdir
        results_dir.mkdir(parents=True, exist_ok=True)

        out_path = results_dir / "results_mi_uni_test.csv"
        mi_results.to_csv(out_path, index=False)
        logger.info("Saved MI results to: %s", out_path)

        top_k = (
            mi_results.head(args.k)[["Protein"]]
            .rename(columns={"Protein": "protein"})
        )
        features_out = results_dir / f"selected_features_k{args.k}.csv"
        top_k.to_csv(features_out, index=False)
        logger.info("Saved top-%d features to: %s", args.k, features_out)

    logger.info("Finished in %.2f s", time.time() - start)


if __name__ == "__main__":
    main()