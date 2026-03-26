# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import logging
import re
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import umap.umap_ as umap

from scipy import stats
from statsmodels.stats.multitest import multipletests
#from tqdm import tqdm

from styles.colors import get_colors
from src.core.data_utils import load_annotation, load_data
from src.core.logging_utils import setup_logging

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOG_SUBDIR = "protein_sub_selection"
N_ITER = 2500

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def run_ttest(X, y, p_threshold=0.05, proteins=None):
    """Run t-tests for each protein and return the indices of selected proteins."""
    _, p_vals = stats.ttest_ind(
        X[y == 1], 
        X[y == 0], 
        axis=0, 
        equal_var=False, 
        nan_policy='omit'
    )

    # Adjust for multiple testing
    reject, adj_p_vals, _, _ = multipletests(p_vals, alpha=p_threshold, method='fdr_bh')

    if proteins is not None:
        sig_proteins = [prot for prot, is_sig in zip(proteins, reject) if is_sig]
        return reject, adj_p_vals, sig_proteins

    return reject, adj_p_vals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Subsample data and do protein selection."
    )
    parser.add_argument(
        "--subsample_size",
        type=float,
        default=0.8,
        help="Proportion of samples to include in each subsample (default: 0.8).",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save outputs to disk and log to file; otherwise log to terminal and show plot interactively.",
    )
    args = parser.parse_args()

    ## Setup
    logger = setup_logging(args.save_results, LOG_SUBDIR, "protein_sub_selection") 

    ## Load filtered data
    df = load_data("data/processed/filtered_data.csv", logger)

    proteins = [c for c in df.columns if c.startswith("seq.")]
    X = df[proteins]
    y = df["ards"]

    n_samples = len(y)

    # If we want the size of classes to be balanced in the bootstrap samples
    p_sample = (y==1) * (y==0).sum() + (y==0) * (y==1).sum()
    p_sample = p_sample / p_sample.sum()

    # Keep track of which proteins are significant in each iteration
    significant_proteins = {prot: [] for prot in proteins}
    p_values = {prot: [] for prot in proteins}

    for i in range(N_ITER):
        if (i+1) % 100 == 0:
            logger.info(f"Iteration {i+1}/{N_ITER}")
        # Bootstrap sampling or subsampling with replacement
        #boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)

        ## subsample without replacement
        rng = np.random.default_rng(seed=42+i)
        sample_idx = np.random.choice(n_samples, size=int(n_samples*args.subsample_size), replace=False) 
                
        X_sample = X.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]

        # Run t-tests and get selected proteins
        reject, adj_p_vals, sig_proteins = run_ttest(X_sample, y_sample, p_threshold=0.05, proteins=proteins)

        for prot in significant_proteins.keys():
            significant_proteins[prot].append(prot in sig_proteins)
            p_values[prot].append(adj_p_vals[proteins.index(prot)])

    # Save results
    if args.save_results:
        results_dir = Path("results") / "protein_sub_selection"
        results_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(significant_proteins).to_csv(results_dir / f"significant_proteins_{args.subsample_size}.csv", index=False)
        pd.DataFrame(p_values).to_csv(results_dir / f"p_values_{args.subsample_size}.csv", index=False)
        logger.info(f"Results saved to {results_dir}")