
# 1. Hierarchical clustering
# Distance metrics: 1-correlation, mutual information, 

# 2. K-means clustering

# 3. Spectral clustering


# Election of representative protein from each cluster

# Paramaters: 
#           csv or df of proteins and label 
#           method for distance metric (correlation, mutual information)
#           method for clustering (hierarchical, k-means, spectral)
#           method for choosing representative protein
# Output:
#           Df of features and cluster assignments
#           Clustered heatmap of features, with clusters colored by method importance
#           Representative protein for each cluster
#           Graph network visualization of clusters and representative proteins
#           Meta proteins of each cluster 

# OBS: Fix a random seed 
# OBS: good practice to run a sensitivity analysis across a range of τ
# ex τ values (e.g. 0.5, 0.6, 0.7, 0.8) and report how stable the number and composition of communities are.
"""
protein_clustering.py

Protein co-regulation analysis via correlation-based clustering.

Pipeline:
    1. Build weighted correlation graph (threshold on pairwise |r|)
    2. Louvain community detection on the graph
    3. Hierarchical clustering (robustness comparison)
    4. Stratified correlation analysis (ARDS vs non-ARDS)
    5. Permutation tests for ALL candidate pairs (mandatory for valid inference)
    6. Elect representative protein per cluster
    7. Build meta-proteins via within-cluster aggregation
    8. Threshold sensitivity analysis
    9. Visualise: clustered heatmap, network graph
"""

import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from itertools import combinations
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from collections import defaultdict


# =============================================================
# 1. LOAD & PREPROCESS
# =============================================================

def load_data(
    data: str | pd.DataFrame,
    label_col: str = "ards",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load protein expression data from a CSV path or a pre-loaded DataFrame.
    Rows = samples, columns = proteins + outcome label.

    Parameters
    ----------
    data : str or pd.DataFrame
        Either a file path to a CSV, or an already-loaded DataFrame.
    label_col : str
        Name of the outcome column. Default = 'ards'.

    Returns
    -------
    X : pd.DataFrame  —  protein expression matrix (samples x proteins)
    y : pd.Series     —  outcome labels
    """
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError(
            f"Expected a file path (str) or DataFrame, got {type(data)}."
        )

    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found in data. "
            f"Available columns: {df.columns.tolist()}"
        )

    return df.drop(columns=[label_col]), df[label_col]


def preprocess(X: pd.DataFrame, standardize: bool = True) -> pd.DataFrame:
    """
    Optionally standardize to zero mean and unit variance per protein.

    Standardization is recommended before correlation-based analysis to
    prevent scale effects. Note: if standardize=True here, the 'zscore_mean'
    aggregation in build_meta_proteins will operate on already-standardized
    values — this is intentional, as zscore_mean then captures relative
    deviation from the standardized baseline rather than raw expression.

    Parameters
    ----------
    X : pd.DataFrame
        Raw protein expression matrix.
    standardize : bool
        If False, returns X unchanged (e.g. if already normalized upstream).
    """
    if standardize:
        return pd.DataFrame(
            StandardScaler().fit_transform(X), columns=X.columns, index=X.index
        )
    return X


# =============================================================
# 2. CORRELATION GRAPH
# =============================================================

def build_correlation_graph(
    X: pd.DataFrame,
    threshold: float = 0.7,
    method: str = "pearson",
):
    """
    Nodes = proteins. Edges = |r| >= threshold, weight = r.
    Pearson assumes linearity; use Spearman for robustness to outliers.

    Parameters
    ----------
    X : pd.DataFrame
        Standardized protein expression matrix.
    threshold : float
        Minimum absolute correlation to include an edge.
        See threshold_sensitivity() to guide this choice.
    method : str
        'pearson' or 'spearman'.

    Returns
    -------
    G : nx.Graph            — weighted correlation graph
    corr : pd.DataFrame     — full pairwise correlation matrix
    """
    corr = X.corr(method=method)
    G = nx.Graph()
    G.add_nodes_from(X.columns)
    proteins = corr.columns.tolist()
    for i, p1 in enumerate(proteins):
        for p2 in proteins[i+1:]:
            c = corr.loc[p1, p2]
            if abs(c) >= threshold:
                G.add_edge(p1, p2, weight=c)
    return G, corr


# =============================================================
# 3. LOUVAIN COMMUNITY DETECTION
# =============================================================

def louvain_clustering(G: nx.Graph) -> dict:
    """
    Partition the correlation graph via Louvain modularity optimisation.
    Uses NetworkX built-in implementation (requires networkx >= 2.7).
    No k required. Returns {protein: cluster_id}.
    """
    communities = nx.community.louvain_communities(G, weight="weight", seed=42)
    return {protein: cid
            for cid, community in enumerate(communities)
            for protein in community}


# =============================================================
# 4. HIERARCHICAL CLUSTERING  (robustness check vs Louvain)
# =============================================================

def hierarchical_clustering(
    corr: pd.DataFrame,
    n_clusters: int = None,
    linkage_method: str = "average",
) -> pd.Series:
    """
    Distance = 1 - |r|. If n_clusters given, returns flat labels.
    Otherwise plots dendrogram for visual inspection of cluster count.

    Used as a robustness check: clusters should broadly agree with Louvain.
    Disagreements may indicate sensitivity to threshold or graph sparsity.
    """
    dist = squareform(1 - corr.abs().values, checks=False)
    Z = linkage(dist, method=linkage_method)
    if n_clusters:
        return pd.Series(
            fcluster(Z, t=n_clusters, criterion="maxclust"),
            index=corr.columns,
            name="hc_cluster",
        )
    # No n_clusters given — plot dendrogram for visual inspection
    plt.figure(figsize=(14, 5))
    dendrogram(Z, labels=corr.columns.tolist(), leaf_rotation=90)
    plt.title("Dendrogram  (distance = 1 − |r|)")
    plt.tight_layout()
    plt.show()
    return pd.Series(dtype=int)


# =============================================================
# 5. STRATIFIED CORRELATION  (ARDS vs non-ARDS)
# =============================================================

def stratified_correlation(
    X: pd.DataFrame,
    y: pd.Series,
    pos=1,
    neg=0,
) -> pd.DataFrame:
    """
    Returns difference matrix (corr_ARDS - corr_nonARDS).
    Large differences flag co-regulation changes specific to disease.
    """
    return X[y == pos].corr() - X[y == neg].corr()


def permutation_test_corr_diff(
    X: pd.DataFrame,
    y: pd.Series,
    n_perm: int = 10000,
    diff_threshold: float = 0.3,
    fdr_alpha: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Runs permutation tests for ALL protein pairs where |corr diff| >= threshold.

    This must be run on all candidate pairs together — not on a manually
    selected subset — to avoid selection bias. Selecting pairs based on
    observed differences and then testing only those would inflate significance,
    as large differences are more likely to appear by chance when many pairs
    are examined. Running tests on all pairs above the threshold and applying
    BH correction controls the false discovery rate across all comparisons.

    Parameters
    ----------
    X : pd.DataFrame
        Protein expression matrix.
    y : pd.Series
        Outcome labels (1=ARDS, 0=non-ARDS).
    n_perm : int
        Number of permutations per test. 1000 is a reasonable default;
        increase to 5000+ for final results.
    diff_threshold : float
        Minimum |corr diff| to include a pair. Filters uninteresting pairs
        and reduces computation time. Use threshold_sensitivity() to guide
        this choice. Note: this is a pre-filter, not a significance criterion
        — all pairs above it are tested and corrected together.
    fdr_alpha : float
        Significance level after Benjamini-Hochberg correction.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    results : pd.DataFrame with columns:
        protein_1, protein_2, obs_diff, p_value, p_adjusted, significant
    """
    rng = np.random.default_rng(seed)

    # --- Compute difference matrix for all pairs ---
    corr_pos = X[y == 1].corr()
    corr_neg = X[y == 0].corr()
    diff_matrix = corr_pos - corr_neg

    # --- Filter pairs above threshold ---
    proteins = X.columns.tolist()
    candidate_pairs = [
        (p1, p2)
        for p1, p2 in combinations(proteins, 2)
        if abs(diff_matrix.loc[p1, p2]) >= diff_threshold
    ]

    if not candidate_pairs:
        print(f"No pairs with |diff| >= {diff_threshold} found. "
              f"Consider lowering diff_threshold.")
        return pd.DataFrame()

    print(f"Running permutation tests for {len(candidate_pairs)} candidate pairs "
          f"({n_perm} permutations each)...")

    # --- Permutation test for each candidate pair ---
    observed_diffs = []
    p_values = []

    for p1, p2 in candidate_pairs:
        obs = diff_matrix.loc[p1, p2]

        null = []
        for _ in range(n_perm):
            yp = rng.permutation(y.values)
            d = (X.loc[yp == 1, [p1, p2]].corr().iloc[0, 1]
               - X.loc[yp == 0, [p1, p2]].corr().iloc[0, 1])
            null.append(d)

        p_val = float(np.mean(np.abs(null) >= abs(obs)))
        observed_diffs.append(obs)
        p_values.append(p_val)

    # --- Benjamini-Hochberg correction across ALL tested pairs ---
    _, p_adjusted, _, _ = multipletests(p_values, alpha=fdr_alpha, method="fdr_bh")

    # --- Build results DataFrame ---
    results = pd.DataFrame({
        "protein_1":   [pair[0] for pair in candidate_pairs],
        "protein_2":   [pair[1] for pair in candidate_pairs],
        "obs_diff":    observed_diffs,
        "p_value":     p_values,
        "p_adjusted":  p_adjusted,
        "significant": p_adjusted < fdr_alpha,
    }).sort_values("p_adjusted").reset_index(drop=True)

    n_sig = results["significant"].sum()
    print(f"Done. {n_sig}/{len(candidate_pairs)} pairs significant at FDR={fdr_alpha}.")

    return results


# =============================================================
# 6. REPRESENTATIVE PROTEIN PER CLUSTER
# =============================================================

def elect_representative(
    X: pd.DataFrame,
    proteins: list,
    G: nx.Graph,
    y: pd.Series = None,
    method: str = "connectivity",
) -> str:
    """
    Select a single representative protein for a cluster.

    Parameters
    ----------
    method : str
        'connectivity' : highest weighted degree in cluster subgraph.
                         Best for network analysis and biological interpretation.
        'outcome'      : highest |corr| with outcome y.
                         Best for predictive models and clinical relevance.
        'stability'    : lowest coefficient of variation across samples.
                         Best for robustness and reproducibility.
    """
    sg = G.subgraph(proteins)
    if method == "connectivity":
        return max(
            proteins,
            key=lambda p: sg.degree(p, weight="weight")
        )
    if method == "outcome":
        assert y is not None, "y required for method='outcome'."
        return max(proteins, key=lambda p: abs(X[p].corr(y)))
    if method == "stability":
        return min(
            proteins,
            key=lambda p: X[p].std() / (abs(X[p].mean()) + 1e-8)
        )
    raise ValueError(
        f"Unknown method: '{method}'. "
        f"Choose from: 'connectivity', 'outcome', 'stability'."
    )


# =============================================================
# 7. META-PROTEINS
# =============================================================

def build_meta_proteins(
    X: pd.DataFrame,
    partition: dict,
    y: pd.Series = None,
    aggregation: str = "zscore_mean",
) -> pd.DataFrame:
    """
    Collapse each cluster into a single meta-protein feature.

    Parameters
    ----------
    X : pd.DataFrame
        Protein expression matrix.
    partition : dict
        {protein: cluster_id} from Louvain.
    y : pd.Series or None
        Outcome labels — required for 'weighted_mean'.
    aggregation : str
        'zscore_mean'   : mean of within-cluster z-scores. Recommended for
                          interaction analysis and cross-cluster comparability.
        'weighted_mean' : weighted mean where proteins with stronger
                          correlation to outcome y have higher weight.
                          Requires y. Best when clinical relevance is priority.
        'mean'          : simple mean of raw expression values.
                          Simple but sensitive to scale differences.
        'median'        : median of raw expression values.
                          More robust to outliers than mean.
        'pca_1'         : first principal component. Captures maximum shared
                          variance but is hard to interpret and compare
                          directly against other proteins.

    Returns
    -------
    meta_df : pd.DataFrame — samples x n_clusters, one column per meta-protein.
    """
    from sklearn.decomposition import PCA

    if aggregation == "weighted_mean" and y is None:
        raise ValueError("y is required for aggregation='weighted_mean'.")

    # Invert partition: {cluster_id: [proteins]}
    groups = defaultdict(list)
    for protein, cid in partition.items():
        groups[cid].append(protein)

    meta = {}
    for cid, proteins in groups.items():
        data = X[proteins]

        if aggregation == "zscore_mean":
            z = (data - data.mean()) / data.std()
            meta[f"meta_{cid}"] = z.mean(axis=1)

        elif aggregation == "weighted_mean":
            weights = data.corrwith(y).abs()
            if weights.sum() == 0:
                # Fallback: equal weights if no protein correlates with outcome
                weights = pd.Series(np.ones(len(proteins)), index=proteins)
            else:
                weights = weights / weights.sum()
            meta[f"meta_{cid}"] = data.dot(weights)

        elif aggregation == "mean":
            meta[f"meta_{cid}"] = data.mean(axis=1)

        elif aggregation == "median":
            meta[f"meta_{cid}"] = data.median(axis=1)

        elif aggregation == "pca_1":
            meta[f"meta_{cid}"] = PCA(1).fit_transform(data).flatten()

        else:
            raise ValueError(
                f"Unknown aggregation: '{aggregation}'. "
                f"Choose from: 'zscore_mean', 'weighted_mean', 'mean', 'median', 'pca_1'."
            )

    return pd.DataFrame(meta, index=X.index)


# =============================================================
# 8. THRESHOLD SENSITIVITY
# =============================================================

def threshold_sensitivity(
    X: pd.DataFrame,
    y: pd.Series,
    thresholds: list = [0.5, 0.6, 0.7, 0.8],
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Evaluate how the choice of correlation threshold affects the clustering.

    Runs Louvain clustering at each threshold value and reports the number
    of resulting communities, edges, and isolated nodes (proteins that fall
    below the threshold for all pairs and are excluded from all communities).
    Use this to guide the choice of threshold before running the full pipeline.

    Parameters
    ----------
    thresholds : list of float
        Threshold values to evaluate.
    method : str
        Correlation method ('pearson' or 'spearman').

    Returns
    -------
    summary : pd.DataFrame with columns:
        threshold, n_clusters, n_edges, n_isolated, modularity
    """
    rows = []
    for t in thresholds:
        G, _ = build_correlation_graph(X, threshold=t, method=method)
        partition = louvain_clustering(G)

        n_clusters = len(set(partition.values()))
        n_edges    = G.number_of_edges()
        n_isolated = sum(1 for n in G.nodes() if G.degree(n) == 0)
        modularity = nx.community.modularity(
            G, [
                {p for p, c in partition.items() if c == cid}
                for cid in set(partition.values())
            ]
        )

        rows.append({
            "threshold":  t,
            "n_clusters": n_clusters,
            "n_edges":    n_edges,
            "n_isolated": n_isolated,
            "modularity": round(modularity, 3),
        })
        print(f"threshold={t}  →  {n_clusters} clusters, "
              f"{n_edges} edges, {n_isolated} isolated, "
              f"modularity={modularity:.3f}")

    return pd.DataFrame(rows)


# =============================================================
# 9. VISUALISATION
# =============================================================

def plot_heatmap(corr: pd.DataFrame, partition: dict) -> None:
    """Clustered correlation heatmap, proteins ordered and coloured by community."""
    order = pd.Series(partition).sort_values()
    c = corr.loc[order.index, order.index]
    palette = sns.color_palette("tab10", order.nunique())
    colors = order.map(
        {cid: palette[i] for i, cid in enumerate(sorted(order.unique()))}
    )
    sns.clustermap(
        c,
        row_colors=colors, col_colors=colors,
        cmap="coolwarm", vmin=-1, vmax=1,
        row_cluster=False, col_cluster=False,
        figsize=(12, 12),
    )
    plt.suptitle("Protein co-regulation heatmap (Louvain clusters)", y=1.01)
    plt.show()


def plot_network(
    G: nx.Graph,
    partition: dict,
    representatives: dict = None,
) -> None:
    """
    Spring-layout network; nodes coloured by community, representatives enlarged.
    Only representative proteins are labelled to avoid overcrowding.
    """
    palette = sns.color_palette("tab10", len(set(partition.values())))
    colors  = [palette[partition[n]] for n in G.nodes()]
    sizes   = [400 if (representatives and n in representatives.values())
               else 80 for n in G.nodes()]

    # Only label representative proteins
    labels = {
        n: n if (representatives and n in representatives.values()) else ""
        for n in G.nodes()
    }

    pos     = nx.spring_layout(G, seed=42, weight="weight")
    weights = [G[u][v]["weight"] for u, v in G.edges()]

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=0.85)
    nx.draw_networkx_edges(G, pos, width=[w*1.5 for w in weights],
                           alpha=0.4, edge_color="grey")
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)
    plt.title("Protein co-regulation network (Louvain communities)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# =============================================================
# 10. MAIN PIPELINE
# =============================================================

def run_pipeline(
    data,
    label_col: str = "ards",
    corr_method: str = "pearson",
    threshold: float = 0.7,
    rep_method: str = "connectivity",
    aggregation: str = "zscore_mean",
    standardize: bool = True,
    n_perm: int = 1000,
    diff_threshold: float = 0.3,
    fdr_alpha: float = 0.05,
    plot: bool = True,
):
    """
    Full pipeline from CSV or DataFrame to cluster assignments,
    representatives, meta-proteins, permutation tests, and visualisations.

    Parameters
    ----------
    data : str or pd.DataFrame
        Path to CSV or a pre-loaded DataFrame.
    label_col : str
        Name of the outcome column.
    corr_method : str
        'pearson' or 'spearman'.
    threshold : float
        Minimum |correlation| to include an edge. Use threshold_sensitivity()
        to guide this choice before running the full pipeline.
    rep_method : str
        'connectivity', 'outcome', or 'stability'.
    aggregation : str
        'zscore_mean', 'weighted_mean', 'mean', 'median', or 'pca_1'.
    standardize : bool
        Whether to standardize X before analysis. Set False if data is
        already normalized upstream.
    n_perm : int
        Number of permutations for significance testing. Use 1000 for
        exploration, 5000+ for final results.
    diff_threshold : float
        Minimum |corr diff| for a protein pair to be permutation tested.
        All pairs above this are tested together to avoid selection bias.
    fdr_alpha : float
        FDR significance level for Benjamini-Hochberg correction.
    plot : bool
        If True, renders heatmap and network graph.

    Returns
    -------
    dict with keys:
        'cluster_assignments' : pd.Series    — protein to Louvain cluster id
        'hc_labels'           : pd.Series    — hierarchical clustering labels
        'representatives'     : dict         — cluster_id to protein name
        'meta_proteins'       : pd.DataFrame — samples x meta-protein features
        'corr_matrix'         : pd.DataFrame — full pairwise correlation matrix
        'diff_matrix'         : pd.DataFrame — stratified correlation differences
        'perm_results'        : pd.DataFrame — permutation test results with FDR
        'graph'               : nx.Graph     — weighted correlation graph
    """
    # --- Load and preprocess ---
    X, y = load_data(data, label_col)
    X = preprocess(X, standardize=standardize)

    # --- Correlation graph and Louvain (primary clustering) ---
    G, corr   = build_correlation_graph(X, threshold, corr_method)
    partition = louvain_clustering(G)

    # --- Hierarchical clustering (robustness check) ---
    hc_labels = hierarchical_clustering(
        corr, n_clusters=len(set(partition.values()))
    )

    # --- Stratified correlation + permutation tests ---
    # Run on ALL pairs above diff_threshold to avoid selection bias.
    # BH correction is applied across all tested pairs simultaneously.
    diff_matrix  = stratified_correlation(X, y)
    perm_results = permutation_test_corr_diff(
        X, y,
        n_perm=n_perm,
        diff_threshold=diff_threshold,
        fdr_alpha=fdr_alpha,
    )

    # --- Invert partition: {cluster_id: [proteins]} ---
    groups = defaultdict(list)
    for p, cid in partition.items():
        groups[cid].append(p)

    # --- Representative proteins ---
    reps = {
        cid: elect_representative(X, ps, G, y=y, method=rep_method)
        for cid, ps in groups.items()
    }

    # --- Meta-proteins ---
    meta = build_meta_proteins(X, partition, y=y, aggregation=aggregation)

    # --- Visualisations ---
    if plot:
        plot_heatmap(corr, partition)
        plot_network(G, partition, representatives=reps)

    return {
        "cluster_assignments": pd.Series(partition, name="louvain_cluster"),
        "hc_labels":           hc_labels,
        "representatives":     reps,
        "meta_proteins":       meta,
        "corr_matrix":         corr,
        "diff_matrix":         diff_matrix,
        "perm_results":        perm_results,
        "graph":               G,
    }


if __name__ == "__main__":
    # Rekommenderat första steg: kör sensitivity analysis för att välja threshold
    # X, y = load_data("proteins.csv")
    # X = preprocess(X)
    # threshold_sensitivity(X, y, thresholds=[0.5, 0.6, 0.7, 0.8])

    # Kör pipeline med defaults
    results = run_pipeline("proteins.csv")

    # Plocka ut resultat
    # meta_proteins = results["meta_proteins"]   # till prediktiv modell
    # perm_results  = results["perm_results"]    # signifikanta proteinpar
    # representatives = results["representatives"]  # till rapport