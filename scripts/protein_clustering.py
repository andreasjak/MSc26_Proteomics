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
from tqdm import tqdm

from styles.colors import get_colors
from src.core.data_utils import load_annotation, load_data
from src.core.logging_utils import setup_logging

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOG_SUBDIR = "protein_clustering"
SIG_PROTEINS_FILE = "results/protein_sub_selection/significant_proteins_0.8.csv"
P_VALUES_FILE = "results/protein_sub_selection/p_values_0.8.csv"

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def run_kmeans_clustering(X, n_clusters=4):
    """Run KMeans clustering and return cluster labels."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    return cluster_labels

def plot_umap(X, cluster_labels, metric="cosine"):
    """Plot UMAP embedding colored by cluster labels."""
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric=metric, random_state=42)
    embedding = reducer.fit_transform(X)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7)
    plt.title("UMAP Projection of Clusters")
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.grid(True, alpha=0.3)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    #plt.show()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster data based on proteins selected in each iteration."
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save outputs to disk and log to file; otherwise log to terminal and show plot interactively.",
    )
    args = parser.parse_args()

    ## Setup
    logger = setup_logging(args.save_results, LOG_SUBDIR, "pipeline") # CHANGE pipeline name

    ## Load filtered data
    df_sig = pd.read_csv(SIG_PROTEINS_FILE)
    df_pvals = pd.read_csv(P_VALUES_FILE)

    proteins = [c for c in df_sig.columns if c.startswith("seq.")]

    #number of proteins selected in each iteration
    n_selected_proteins = [sum(df_sig[prot][i] for prot in proteins) for i in range(len(df_sig))]
    
    # figure histogram
    plt.figure(figsize=(10, 6))
    plt.hist(n_selected_proteins, bins=50, edgecolor='black')
    plt.title("Distribution of Number of Significant Proteins Across Iterations")
    plt.xlabel("Number of Significant Proteins")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    #plt.show()

    # Reduce dimension by removing proteins that were never significant
    proteins_to_keep = [prot for prot in proteins if np.mean(df_sig[prot]) > 0]
    print(f"Number of proteins to keep: {len(proteins_to_keep)}")

    idx_keep = [proteins.index(prot) for prot in proteins_to_keep]

    significant_proteins_keep = {prot: df_sig[prot] for prot in proteins_to_keep}
    p_values_keep = {prot: df_pvals[prot] for prot in proteins_to_keep}


    # CLUSTERING AND UMAP
    X_sig = df_sig[proteins_to_keep].values
    X_pval = df_pvals[proteins_to_keep].values


    # HERE WE CAN RUN DIFFERENT CLUSTERING METHODS AND PLOT UMAPS FOR BOTH SIGNIFICANT PROTEINS AND P-VALUES, THEN COMPARE THE CLUSTERING QUALITY
    
    # Run KMeans clustering on significant proteins
    cluster_labels_sig = run_kmeans_clustering(X_sig, n_clusters=3)

    # Plot UMAP for significant proteins
    plot_umap(X_sig, cluster_labels_sig, metric="cosine")

    # Evaluate clustering quality
    silhouette_avg = silhouette_score(X_sig, cluster_labels_sig, metric="cosine")
    davies_bouldin = davies_bouldin_score(X_sig, cluster_labels_sig)
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    print(f"Davies-Bouldin Score: {davies_bouldin:.4f}")


    plt.show()