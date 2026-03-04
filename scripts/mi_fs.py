"""
Script to perform interactions information-based feature selection for proteomics data.

Outputs (when --save-results):
  results/<results-subdir>/results_mi_fs.csv
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
from src.core.info_utils import mutual_information_cd, interaction_information_ccd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression



def feature_selection_algo(X, y, n_features, n_neighbors, mi_uni_scores):
    n_total = X.shape[1]
    selected_features = set()
    seleceted_features_orderd = np.array([])
    remaining_features = set(range(n_total))
    ii_cache = {}

    def batch_cache_ii(new_feat, remaining):
        idx = y == 1
        p = np.mean(idx)
        rem_list = list(remaining)
        X_rem = X[:, rem_list]
        x_new = X[:, new_feat]

        mi_joint = mutual_info_regression(X_rem, x_new, n_neighbors=n_neighbors)
        mi_cond1 = mutual_info_regression(X_rem[idx], x_new[idx], n_neighbors=n_neighbors)
        mi_cond0 = mutual_info_regression(X_rem[~idx], x_new[~idx], n_neighbors=n_neighbors)

        ii_vals = mi_joint - p * mi_cond1 - (1 - p) * mi_cond0
        for i, feat in enumerate(rem_list):
            key = (min(new_feat, feat), max(new_feat, feat))
            ii_cache[key] = ii_vals[i]

    # Select first feature, then batch-compute its II against all others
    first_feature = np.argmax(mi_uni_scores)
    selected_features.add(first_feature)
    seleceted_features_orderd = np.append(seleceted_features_orderd, first_feature)
    remaining_features.remove(first_feature)
    batch_cache_ii(first_feature, remaining_features)

    while len(selected_features) < n_features and remaining_features:
        best_feature = None
        best_score = -np.inf
        print(len(selected_features))

        for feature in remaining_features:
            J = mi_uni_scores[feature]
            for selected_feature in selected_features:
                key = (min(feature, selected_feature), max(feature, selected_feature))
                J -= ii_cache[key]

            if J > best_score:
                best_score = J
                best_feature = feature

        selected_features.add(best_feature)
        remaining_features.remove(best_feature)
        # Batch-compute II between new selection and all remaining
        batch_cache_ii(best_feature, remaining_features)
        seleceted_features_orderd = np.append(seleceted_features_orderd, best_feature)

    return seleceted_features_orderd.astype(int).tolist()

def feature_selection_algo_old(
    X,
    y,
    n_features,
    n_neighbors,
    mi_uni_scores
):
    """
    Perform feature selection based on mutual information and interaction information.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (samples x features).
    y : np.ndarray
        Target vector (samples,).
    n_features : int
        Number of features to select.
    n_neighbors : int, optional
        Number of neighbors for MI estimation (default: 3).
    mi_uni_scores : np.ndarray
        Precomputed univariate mutual information scores for each feature.
    """
    n_total = X.shape[1]
    selected_features = set()
    remaining_features = set(range(n_total))
    ii_cache = {}  # (i, j) -> interaction information, i < j

    first_feature = np.argmax(mi_uni_scores)
    selected_features.add(first_feature)
    remaining_features.remove(first_feature)

    while len(selected_features) < n_features and remaining_features:
        best_feature = None
        best_score = -np.inf

        for feature in remaining_features:
            J = mi_uni_scores[feature]

            for selected_feature in selected_features:
                key = (min(feature, selected_feature), max(feature, selected_feature))
                if key not in ii_cache:
                    ii_cache[key] = interaction_information_ccd(
                        X[:, feature], X[:, selected_feature], y,
                        k=n_neighbors, method="sklearn"
                    )
                J -= ii_cache[key]

            if J > best_score:
                best_score = J
                best_feature = feature

        selected_features.add(best_feature)
        remaining_features.remove(best_feature)

    return list(selected_features)

    


def main():
    start = time.time()
    parser = argparse.ArgumentParser(
        description="Features selection based on interaction information."
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
        default="mi_fs",
        help="Subdirectory under results/ for output files (default: mi_fs).",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save outputs to disk and log to file; otherwise log to terminal only.",
    )
    parser.add_argument(
        "--log-subdir",
        type=str,
        default="mi_fs",
        help="Subdirectory under logs/ for log files (default: mi_fs).",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=3,
        help="Number of neighbors for MI estimation (default: 3).",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=20,
        help="Number of proteins in feature subset to save as selected features (default: 20).",
    )

    args = parser.parse_args()

    logger = setup_logging(args.save_results, args.log_subdir, "mi_fs")

    logger.info("Starting mi_fs.py")
    logger.info(
        "Args: data_path=%s  results_subdir=%s  n_features=%d  save_results=%s",
        args.data_path, args.results_subdir, args.n_features, args.save_results,
    )

    # Step 1: Load data
    data = load_data(args.data_path, logger)

    # Step 2: Extract features and labels
    protein_cols = get_protein_features(data)
    logger.info("Protein columns to score: %d", len(protein_cols))
    X = data[protein_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    y = data["ards"].astype(int).values

    # Step 3: Compute univariate mutual information for each protein
    logger.info("Computing univariate MI (k=%d) …", args.n_neighbors)
    #mi_uni_scores = mutual_information_cd(X.values, y, k=args.n_neighbors, method="sklearn")
    mi_uni_scores = mutual_info_classif(X.values, y, n_neighbors=args.n_neighbors)

    # Step 4: Perform feature selection based on interaction information
    logger.info("Performing feature selection based on interaction information (n_features=%d) …", args.n_features)
    selected_indices = feature_selection_algo(X.values, y, n_features=args.n_features, n_neighbors=args.n_neighbors, mi_uni_scores=mi_uni_scores)
    selected_proteins = [protein_cols[i] for i in selected_indices]
    selected_proteins_df = pd.DataFrame(selected_proteins, columns=["protein"])

    # Step 5: Save results
    if args.save_results:
        results_dir = Path("results") / args.results_subdir
        results_dir.mkdir(parents=True, exist_ok=True)

        features_out = results_dir / f"selected_features_n_features{args.n_features}.csv"
        selected_proteins_df.to_csv(features_out, index=False)
        logger.info("Saved top-%d features to: %s", args.n_features, features_out)
    else:
        logger.info("Selected features:\n%s", selected_proteins_df)

    logger.info("Finished in %.2f s", time.time() - start)

if __name__ == "__main__":
    main()