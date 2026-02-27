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
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.multitest import multipletests

from src.core.data_utils import get_protein_features


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(
    save_results: bool, log_subdir: str, script_name: str
) -> logging.Logger:
    """
    Configure logging:
      - save_results=False → StreamHandler (terminal) only
      - save_results=True  → FileHandler (file) only, no terminal output
    Log path: logs/<log_subdir>/<script_name>_YYYYMMDD_HHMMSS.log
    """
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if save_results:
        log_dir = Path("logs") / log_subdir
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"{script_name}_{timestamp}.log"
        handler: logging.Handler = logging.FileHandler(log_path)
    else:
        handler = logging.StreamHandler()

    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def load_data(
    data_path: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Load seen.csv and return the full DataFrame."""
    logger.info("Loading data from %s", data_path)
    data = pd.read_csv(data_path)
    logger.info("Data shape: %s", data.shape)

    n_ards = int((data["ards"] == 1).sum())
    n_non = int((data["ards"] == 0).sum())
    logger.info("Samples — ARDS: %d | non-ARDS: %d", n_ards, n_non)

    return data


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

## ---------------------------------------------------------------------------
## WE SHOULD DO MONTE CARLO PERMUTATION TESTING
## ---------------------------------------------------------------------------
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

    _, adj_p, _, _ = multipletests(results["p_perm"].values, method=mt_method)
    results = results.copy()
    results["ADJ_P"] = adj_p
    results = (
        results
        .sort_values(["ADJ_P", "MI"], ascending=[True, False])
        .reset_index(drop=True)
    )

    n_sig = int((results["ADJ_P"] < alpha).sum())
    logger.info(
        "Correction: %s | alpha=%.3f | significant proteins: %d / %d",
        mt_method, alpha, n_sig, len(results),
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
        default="mi",
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
        default="mi",
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
        "--n-perm",
        type=int,
        default=1000,
        help="Number of permutations for empirical p-values (default: 1000).",
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
        default=10,
        help="Number of top proteins to save as selected features (default: 10).",
    )
    args = parser.parse_args()

    logger = setup_logging(args.save_results, args.log_subdir, "mi_uni_tests")

    logger.info("Starting mi_uni_tests.py")
    logger.info(
        "Args: data_path=%s  correction=%s  alpha=%s  n_perm=%d  "
        "n_neighbors=%d  random_state=%d  k=%d  save_results=%s",
        args.data_path, args.correction_method, args.alpha, args.n_perm,
        args.n_neighbors, args.random_state, args.k, args.save_results,
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

    # Step 3: Permutation test
    mi_results = permutation_test(
        X_perm, y, mi_results,
        args.n_perm, args.n_neighbors, args.random_state, logger,
    )

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