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

from styles.colors import get_colors
from src.core.data_utils import load_annotation, load_data
from src.core.logging_utils import setup_logging
from sklearn.metrics.pairwise import pairwise_distances

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOG_SUBDIR = "protein_clustering"
SIG_PROTEINS_FILE = "results/protein_sub_selection/significant_proteins_0.8.csv"
P_VALUES_FILE = "results/protein_sub_selection/p_values_0.8.csv"

# ---------------------------------------------------------------------------
# Utilities - clustering
# ---------------------------------------------------------------------------

def run_kmeans_clustering(X, n_clusters=4):
    """Run KMeans clustering and return cluster labels."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    return cluster_labels

def run_dbscan_clustering(X, eps=0.5, min_samples=10, metric="euclidean"):
    """Run DBSCAN clustering and return cluster labels."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    cluster_labels = dbscan.fit_predict(X)
    return cluster_labels

# ---------------------------------------------------------------------------
# Utilities - plotting
# ---------------------------------------------------------------------------

def plot_umap(X, cluster_labels, metric="cosine", random_state=42):
    """Plot UMAP embedding colored by cluster labels."""
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric=metric, random_state=random_state)
    embedding = reducer.fit_transform(X)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7)
    plt.title("UMAP Projection of Clusters")
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.grid(True, alpha=0.3)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    #plt.show()

def bar_plot_cluster_selection_probabilities(X, cluster_labels, proteins):
    """Bar plot of mean selection probabilities for each protein in each cluster."""
    unique_clusters = np.unique(cluster_labels)
    mean_selection_probs = {cluster: np.mean(X[cluster_labels == cluster], axis=0) for cluster in unique_clusters}

    x = np.arange(len(proteins))
    width = 1 / (len(unique_clusters)+1)

    plt.figure(figsize=(12, 6))
    for i, cluster in enumerate(unique_clusters):
        plt.bar(x + i*width - width/2 * len(unique_clusters), mean_selection_probs[cluster], width, label=f'Cluster {cluster+1 if cluster >= 0 else "Noise"}', alpha=0.7)
    
    plt.xlabel('Proteins')
    plt.ylabel('Mean Selection Probability')
    plt.title('Mean Selection Probability of Proteins in Each Cluster')
    plt.xticks(x, proteins, rotation=90)
    plt.legend()
    plt.tight_layout()
    #plt.show()

# ---------------------------------------------------------------------------
# Utilities - distance metrics (can probably use sklearn's pairwise_distances instead of implementing these ourselves, but let's keep them here for now)
# ---------------------------------------------------------------------------

def jaccard_distance(x, y):
    """Compute Jaccard distance between two binary vectors."""
    intersection = np.sum(np.logical_and(x, y))
    union = np.sum(np.logical_or(x, y))
    if union == 0:
        return 0.0
    return 1 - intersection / union

def hamming_distance(x, y):
    """Compute Hamming distance between two binary vectors."""
    return np.sum(x != y) / len(x)

def cosine_distance(x, y):
    """Compute Cosine distance between two vectors."""
    if np.all(x == 0) or np.all(y == 0):
        return 1.0
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def euclidean_distance(x, y):
    """Compute Euclidean distance between two vectors."""
    return np.linalg.norm(x - y)

def manhattan_distance(x, y):
    """Compute Manhattan distance between two vectors."""
    return np.sum(np.abs(x - y))

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
    logger = setup_logging(args.save_results, LOG_SUBDIR, "protein_clustering")

    ## Load data
    df_sig = pd.read_csv(SIG_PROTEINS_FILE)
    logger.info(f"Loaded significant proteins data from {SIG_PROTEINS_FILE} with shape {df_sig.shape}")
    
    df_pvals = pd.read_csv(P_VALUES_FILE)
    logger.info(f"Loaded p-values data from {P_VALUES_FILE} with shape {df_pvals.shape}")

    proteins = [c for c in df_sig.columns if c.startswith("seq.")]

    #number of proteins selected in each iteration
    n_selected_proteins = df_sig.sum(axis=1).values

    iter_keep = n_selected_proteins >= 0
    logger.info(f"Number of iterations with at least 0 significant proteins: {np.sum(iter_keep)} out of {len(n_selected_proteins)}")

    df_sig_keep = df_sig[iter_keep]
    df_pvals_keep = df_pvals[iter_keep]

    # figure histogram
    plt.figure(figsize=(10, 6))
    plt.hist(n_selected_proteins[iter_keep], bins=50, edgecolor='black')
    plt.title("Distribution of Number of Significant Proteins Across Iterations")
    plt.xlabel("Number of Significant Proteins")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    #plt.show()

    # Reduce dimension by removing proteins that were never significant
    proteins_to_keep = [prot for prot in proteins if np.mean(df_sig_keep[prot]) > 0.1]
    logger.info(f"Number of proteins to keep: {len(proteins_to_keep)}")

    idx_keep = [proteins.index(prot) for prot in proteins_to_keep]

    # CLUSTERING AND UMAP
    X_sig = df_sig_keep[proteins_to_keep].values
    X_pval = df_pvals_keep[proteins_to_keep].values


    # HERE WE CAN RUN DIFFERENT CLUSTERING METHODS AND PLOT UMAPS FOR BOTH SIGNIFICANT PROTEINS AND P-VALUES, THEN COMPARE THE CLUSTERING QUALITY
    #cluster_labels = run_dbscan_clustering(-np.log(X_pval), eps=0.5, min_samples=15, metric="euclidean")
    cluster_labels = run_kmeans_clustering(-np.log10(X_pval), n_clusters=4)
    silhouette_avg = silhouette_score(-np.log10(X_pval), cluster_labels, metric="euclidean")
    davies_bouldin = davies_bouldin_score(-np.log10(X_pval), cluster_labels)
    logger.info(f"Clustering - Silhouette Score: {silhouette_avg:.4f}, Davies-Bouldin Score: {davies_bouldin:.4f}")

    # log cluster sizes
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    for cluster, count in zip(unique_clusters, counts):
        logger.info(f"Cluster {cluster}: {count} iterations")

    # Plot UMAP for significant proteins
    plot_umap(-np.log10(X_pval), cluster_labels, metric="euclidean", random_state=None)

    bar_plot_cluster_selection_probabilities(X_sig, cluster_labels, proteins_to_keep)

    plt.show()

'''
    best_silhouette = -1
    best_db_score = np.inf
    best_k = None

    for k in [2, 3, 4, 5, 6]:
        # Run KMeans clustering on p-values
        cluster_labels = run_kmeans_clustering(X_pval, n_clusters=k)
        silhouette_avg = silhouette_score(X_pval, cluster_labels, metric="euclidean")
        davies_bouldin = davies_bouldin_score(X_pval, cluster_labels)

        # Update best scores
        if silhouette_avg > best_silhouette:
            best_silhouette = silhouette_avg
            best_db_score = davies_bouldin
            best_cluster_labels = cluster_labels
            best_k = k

    logger.info(f"Best K: {best_k} with Silhouette Score: {best_silhouette:.4f} and Davies-Bouldin Score: {best_db_score:.4f}")

    # Plot UMAP for significant proteins
    plot_umap(X_pval, best_cluster_labels, metric="euclidean")

    # Evaluate clustering quality
    
    logger.info(f"Silhouette Score: {best_silhouette:.4f}")
    logger.info(f"Davies-Bouldin Score: {best_db_score:.4f}")

    bar_plot_cluster_selection_probabilities(X_sig, best_cluster_labels, proteins_to_keep)

    plt.show()
'''