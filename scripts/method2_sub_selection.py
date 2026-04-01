"""Methodology 2: subsampling-based feature selection and cluster graph building.

This script reproduces the notebook Methodology 2 workflow in a reusable way:
1. Run stratified subsampling selection iterations (t-test, univariate MI, or RF+SHAP selector).
2. Compute selection frequency pi and filter proteins.
3. Build R co-selection matrix and graph network.
4. Detect Louvain communities and return a cluster DataFrame.

Input:
- CSV path OR pandas DataFrame containing protein columns (prefix: seq.) and label column (default: ards)
- Feature selection method name (selector)

Output:
- cluster_df (DataFrame with protein, pi, cluster)
- G (networkx Graph)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import logging
import time
from typing import Any, Callable

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests

from src.core.logging_utils import setup_logging

# ---------------------------------------------------------------------------
# Feature selectors
# ---------------------------------------------------------------------------


def run_ttest_selector(
    X_sub: pd.DataFrame,
    y_sub: pd.Series,
    mode: str = "top_k",
    top_k: int = 20,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Return binary selected mask and adjusted p-values for t-test selector."""
    _, p_vals = stats.ttest_ind(
        X_sub[y_sub == 1],
        X_sub[y_sub == 0],
        axis=0,
        equal_var=False,
        nan_policy="omit",
    )

    p_vals = np.nan_to_num(p_vals, nan=1.0, posinf=1.0, neginf=1.0)
    _, adj_p_vals, _, _ = multipletests(p_vals, alpha=alpha, method="fdr_bh")

    if mode == "top_k":
        k = min(top_k, len(adj_p_vals))
        idx = np.argsort(adj_p_vals)[:k]
        selected = np.zeros(len(adj_p_vals), dtype=int)
        selected[idx] = 1
    elif mode == "fdr":
        selected = (adj_p_vals < alpha).astype(int)
    else:
        raise ValueError("mode must be 'top_k' or 'fdr'")

    return selected, adj_p_vals


SELECTOR_REGISTRY: dict[str, Callable[..., tuple[np.ndarray, np.ndarray]]] = {
    "ttest": run_ttest_selector,
}

def uni_mi_selector(
    X_sub: pd.DataFrame,
    y_sub: pd.Series,
    mode: str = "top_k",
    top_k: int = 20,
    alpha: float = 0.05,
    n_neighbors: int = 3,
    n_perm: int = 0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return selected mask and adjusted p-values for univariate MI selector.

    The core MI computation mirrors `mi_uni_test.py`: numeric coercion, median
    imputation, and `mutual_info_classif` scoring.

    Notes:
    - `mode='top_k'` uses MI ranking directly.
    - `mode='fdr'` requires permutation p-values (`n_perm >= 1`) and applies
      Benjamini-Hochberg correction.
    """
    X_num = X_sub.apply(pd.to_numeric, errors="coerce")
    X_num = X_num.fillna(X_num.median())
    y_arr = y_sub.astype(int).values

    mi_scores = mutual_info_classif(
        X_num.values,
        y_arr,
        discrete_features=False,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )

    n_features = len(mi_scores)
    selected = np.zeros(n_features, dtype=int)

    if mode == "top_k":
        k = min(top_k, n_features)
        idx = np.argsort(mi_scores)[::-1][:k]
        selected[idx] = 1

        # P-values are optional in top_k mode; compute only when requested.
        if n_perm >= 1:
            rng = np.random.default_rng(random_state)
            hits = np.zeros(n_features, dtype=int)
            for b in range(n_perm):
                y_perm = rng.permutation(y_arr)
                mi_perm = mutual_info_classif(
                    X_num.values,
                    y_perm,
                    discrete_features=False,
                    n_neighbors=n_neighbors,
                    random_state=random_state + b + 1,
                )
                hits += (mi_perm >= mi_scores).astype(int)

            p_perm = (hits + 1) / (n_perm + 1)
            _, adj_p_vals, _, _ = multipletests(p_perm, alpha=alpha, method="fdr_bh")
        else:
            adj_p_vals = np.full(n_features, np.nan)

    elif mode == "fdr":
        if n_perm < 1:
            raise ValueError("uni_mi selector with mode='fdr' requires n_perm >= 1")

        rng = np.random.default_rng(random_state)
        hits = np.zeros(n_features, dtype=int)
        for b in range(n_perm):
            y_perm = rng.permutation(y_arr)
            mi_perm = mutual_info_classif(
                X_num.values,
                y_perm,
                discrete_features=False,
                n_neighbors=n_neighbors,
                random_state=random_state + b + 1,
            )
            hits += (mi_perm >= mi_scores).astype(int)

        p_perm = (hits + 1) / (n_perm + 1)
        _, adj_p_vals, _, _ = multipletests(p_perm, alpha=alpha, method="fdr_bh")
        selected = (adj_p_vals < alpha).astype(int)
    else:
        raise ValueError("mode must be 'top_k' or 'fdr'")

    return selected, adj_p_vals

SELECTOR_REGISTRY["uni_mi"] = uni_mi_selector

def rf_shap_selector(
    X_sub: pd.DataFrame,
    y_sub: pd.Series,
    mode: str = "top_k",
    top_k: int = 20,
    alpha: float = 0.05,
    random_state: int = 42,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    val_size: float = 0.3,
    n_repeats: int = 20,
    pos_weight: float = 3.0,
    perm_weight: float = 0.6,
    shap_weight: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Select proteins using RF permutation importance + SHAP class-1 importance.

    ARDS-positive detection is prioritized by using recall for class 1 as the
    permutation-importance scoring target and class weighting in the RF model.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "rf_shap selector requires the 'shap' package. Install it to use this selector."
        ) from exc

    X_num = X_sub.apply(pd.to_numeric, errors="coerce")
    X_num = X_num.fillna(X_num.median())
    y_arr = y_sub.astype(int).values

    X_train, X_val, y_train, y_val = train_test_split(
        X_num,
        y_arr,
        test_size=val_size,
        stratify=y_arr,
        random_state=random_state,
    )

    class_weight = {0: 1.0, 1: float(pos_weight)}
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Permutation importance for ARDS=1 recall (positive class sensitivity).
    scorer = make_scorer(recall_score, pos_label=1)
    perm = permutation_importance(
        rf,
        X_val,
        y_val,
        scoring=scorer,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    perm_mean = np.clip(perm.importances_mean, a_min=0.0, a_max=None)
    perm_p = ((perm.importances <= 0.0).sum(axis=1) + 1) / (n_repeats + 1)

    # SHAP importance for class 1 (ARDS): mean absolute SHAP value.
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X_val)

    if isinstance(shap_vals, list):
        shap_arr = np.asarray(shap_vals[1] if len(shap_vals) > 1 else shap_vals[0])
    else:
        shap_arr = np.asarray(shap_vals)
        if shap_arr.ndim == 3:
            shap_arr = shap_arr[:, :, 1] if shap_arr.shape[2] > 1 else shap_arr[:, :, 0]

    shap_mean_abs = np.mean(np.abs(shap_arr), axis=0)

    def _norm(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        s = float(v.sum())
        if s <= 0:
            return np.zeros_like(v)
        return v / s

    # Combine scaled importance measures; weights are normalized for safety.
    w_perm = max(float(perm_weight), 0.0)
    w_shap = max(float(shap_weight), 0.0)
    w_sum = w_perm + w_shap
    if w_sum == 0:
        w_perm, w_shap = 0.5, 0.5
    else:
        w_perm, w_shap = w_perm / w_sum, w_shap / w_sum

    combined_score = w_perm * _norm(perm_mean) + w_shap * _norm(shap_mean_abs)

    n_features = X_num.shape[1]
    selected = np.zeros(n_features, dtype=int)

    # Keep adjusted p-values from permutation-importance significance.
    _, adj_p_vals, _, _ = multipletests(perm_p, alpha=alpha, method="fdr_bh")

    if mode == "top_k":
        k = min(top_k, n_features)
        idx = np.argsort(combined_score)[::-1][:k]
        selected[idx] = 1
    elif mode == "fdr":
        selected = (adj_p_vals < alpha).astype(int)
    else:
        raise ValueError("mode must be 'top_k' or 'fdr'")

    return selected, adj_p_vals
SELECTOR_REGISTRY["rf_shap"] = rf_shap_selector

def _normalize_selector(selector: str) -> str:
    """Normalize selector names and common aliases to canonical keys."""
    key = str(selector).strip().lower()
    alias_map = {
        "ttest": "ttest",
        "t_test": "ttest",
        "uni_mi": "uni_mi",
        "mi_uni": "uni_mi",
        "mi": "uni_mi",
        "rf_shap": "rf_shap",
        "rd_shap": "rf_shap",
    }
    if key not in alias_map:
        raise ValueError(
            f"Unknown selector: {selector}. Available: {list(SELECTOR_REGISTRY)}"
        )
    return alias_map[key]



# ---------------------------------------------------------------------------
# Core workflow
# ---------------------------------------------------------------------------


def _load_input_data(input_data: pd.DataFrame | str | Path) -> pd.DataFrame:
    """Load input as DataFrame from DataFrame object or CSV path."""
    if isinstance(input_data, pd.DataFrame):
        return input_data.copy()
    return pd.read_csv(input_data)


def run_subsampling_selection(
    X: pd.DataFrame,
    y: pd.Series,
    selector: str = "ttest",
    n_iter: int = 300,
    subsample_size: float = 0.5,
    random_state: int = 42,
    selector_kwargs: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run stratified subsampling and build Z (selection) and P (adj p-values)."""
    selector = _normalize_selector(selector)

    selector_fn = SELECTOR_REGISTRY[selector]
    selector_kwargs = selector_kwargs or {}

    n_samples, n_proteins = X.shape
    Z = np.zeros((n_iter, n_proteins), dtype=int)
    P = np.zeros((n_iter, n_proteins), dtype=float)
    all_idx = np.arange(n_samples)

    for i in range(n_iter):
        sub_idx, _ = train_test_split(
            all_idx,
            train_size=subsample_size,
            stratify=y,
            random_state=random_state + i,
        )

        X_sub = X.iloc[sub_idx]
        y_sub = y.iloc[sub_idx]

        selected, adj_p_vals = selector_fn(X_sub, y_sub, **selector_kwargs)
        Z[i, :] = selected
        P[i, :] = adj_p_vals

        if logger is not None and (i + 1) % 100 == 0:
            logger.info("Iteration %d/%d", i + 1, n_iter)

    return Z, P


def build_graph_and_clusters(
    Z: np.ndarray,
    proteins: list[str],
    pi_thr: float = 0.05,
    edge_thr: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, nx.Graph]:
    """Filter proteins by pi, build co-selection graph, and detect communities."""
    pi = Z.mean(axis=0)
    keep_mask = pi >= pi_thr

    Z_filt = Z[:, keep_mask]
    proteins_filt = np.array(proteins)[keep_mask]
    pi_filt = pi[keep_mask]

    G = nx.Graph()
    for prot in proteins_filt:
        G.add_node(str(prot))

    if len(proteins_filt) == 0:
        return pd.DataFrame(columns=["protein", "pi", "cluster"]), G

    R = np.corrcoef(Z_filt, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)

    p = len(proteins_filt)
    for i in range(p):
        for j in range(i + 1, p):
            w = float(R[i, j])
            if w > edge_thr:
                G.add_edge(str(proteins_filt[i]), str(proteins_filt[j]), weight=w)

    # Handle sparse graphs safely.
    if G.number_of_nodes() == 0:
        communities: list[set[str]] = []
    elif G.number_of_edges() == 0:
        communities = [{n} for n in G.nodes()]
    else:
        communities = nx.community.louvain_communities(G, weight="weight", seed=random_state)

    cluster_map: dict[str, int] = {}
    for k, comm in enumerate(communities):
        for prot in comm:
            cluster_map[prot] = k

    for prot in proteins_filt:
        cluster_map.setdefault(str(prot), -1)

    cluster_df = pd.DataFrame(
        {
            "protein": proteins_filt.astype(str),
            "pi": pi_filt,
            "cluster": [cluster_map[str(p)] for p in proteins_filt],
        }
    ).sort_values(["cluster", "pi"], ascending=[True, False]).reset_index(drop=True)

    return cluster_df, G


def run_method2_sub_selection(
    input_data: pd.DataFrame | str | Path,
    selector: str = "ttest",
    label_col: str = "ards",
    protein_prefix: str = "seq.",
    n_iter: int = 300,
    subsample_size: float = 0.5,
    random_state: int = 42,
    selector_kwargs: dict[str, Any] | None = None,
    pi_thr: float = 0.05,
    edge_thr: float = 0.15,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, nx.Graph]:
    """End-to-end Methodology 2 pipeline returning cluster_df and graph G."""
    selector = _normalize_selector(selector)
    df = _load_input_data(input_data)

    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    proteins = [c for c in df.columns if str(c).startswith(protein_prefix)]
    if len(proteins) == 0:
        raise ValueError(f"No protein columns found with prefix '{protein_prefix}'.")

    X = df[proteins].copy()
    y = df[label_col].astype(int).copy()

    if logger is not None:
        logger.info("Input shape: X=%s, y=%s", X.shape, y.shape)
        logger.info("Selector: %s", selector)

    Z, _ = run_subsampling_selection(
        X=X,
        y=y,
        selector=selector,
        n_iter=n_iter,
        subsample_size=subsample_size,
        random_state=random_state,
        selector_kwargs=selector_kwargs,
        logger=logger,
    )

    cluster_df, G = build_graph_and_clusters(
        Z=Z,
        proteins=proteins,
        pi_thr=pi_thr,
        edge_thr=edge_thr,
        random_state=random_state,
    )

    if logger is not None:
        logger.info("Clusters built: %d proteins, %d graph nodes, %d graph edges",
                    len(cluster_df), G.number_of_nodes(), G.number_of_edges())

    return cluster_df, G


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    start = time.time()

    parser = argparse.ArgumentParser(
        description="Methodology 2 subsampling selection -> graph -> Louvain clusters."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/processed/seen.csv"),
        help="Input CSV path (default: data/processed/seen.csv)",
    )
    parser.add_argument(
        "--selector",
        type=str,
        default="ttest",
        help="Feature selector name (e.g. ttest, uni_mi, rf_shap, rd_shap).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Alias for --selector. If set, it overrides --selector.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="top_k",
        choices=["top_k", "fdr"],
        help="Selector mode (top_k or fdr).",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Top-k features when mode=top_k.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Alpha for FDR mode.")
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=3,
        help="k for mutual_info_classif when selector=uni_mi (default: 3).",
    )
    parser.add_argument(
        "--mi-n-perm",
        type=int,
        default=0,
        help="Permutations for MI p-values when selector=uni_mi (default: 0).",
    )
    parser.add_argument("--rf-n-estimators", type=int, default=300, help="RF trees for selector=rf_shap.")
    parser.add_argument("--rf-max-depth", type=int, default=None, help="RF max depth for selector=rf_shap.")
    parser.add_argument(
        "--rf-min-samples-leaf",
        type=int,
        default=1,
        help="RF min_samples_leaf for selector=rf_shap.",
    )
    parser.add_argument(
        "--rf-val-size",
        type=float,
        default=0.3,
        help="Validation split used by selector=rf_shap (default: 0.3).",
    )
    parser.add_argument(
        "--rf-n-repeats",
        type=int,
        default=20,
        help="Permutation repeats for selector=rf_shap (default: 20).",
    )
    parser.add_argument(
        "--rf-pos-weight",
        type=float,
        default=3.0,
        help="Class weight multiplier for ARDS=1 in selector=rf_shap (default: 3.0).",
    )
    parser.add_argument(
        "--rf-perm-weight",
        type=float,
        default=0.6,
        help="Weight for permutation importance in selector=rf_shap (default: 0.6).",
    )
    parser.add_argument(
        "--rf-shap-weight",
        type=float,
        default=0.4,
        help="Weight for SHAP importance in selector=rf_shap (default: 0.4).",
    )
    parser.add_argument("--n-iter", type=int, default=300, help="Number of subsampling iterations.")
    parser.add_argument("--subsample-size", type=float, default=0.5, help="Subsample fraction per iteration.")
    parser.add_argument("--pi-thr", type=float, default=0.05, help="Selection frequency threshold.")
    parser.add_argument("--edge-thr", type=float, default=0.15, help="Edge threshold for co-selection graph.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--label-col",
        type=str,
        default="ards",
        help="Label column name (default: ards)",
    )
    parser.add_argument(
        "--protein-prefix",
        type=str,
        default="seq.",
        help="Protein column prefix (default: seq.)",
    )
    parser.add_argument(
        "--cluster-out",
        type=Path,
        default=Path("results/method2_sub_selection/cluster_df.csv"),
        help="Output CSV path for cluster DataFrame.",
    )
    parser.add_argument(
        "--graph-out",
        type=Path,
        default=Path("results/method2_sub_selection/graph.graphml"),
        help="Output GraphML path for graph G.",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="If set, save logs to logs/method2_sub_selection/. Otherwise log to terminal.",
    )
    args = parser.parse_args()

    logger = setup_logging(
        save_results=args.save_results,
        log_subdir="method2_sub_selection",
        script_name="method2_sub_selection",
    )

    selector_name = _normalize_selector(args.method if args.method is not None else args.selector)

    if selector_name == "ttest":
        selector_kwargs = {
            "mode": args.mode,
            "top_k": args.top_k,
            "alpha": args.alpha,
        }
    elif selector_name == "uni_mi":
        selector_kwargs = {
            "mode": args.mode,
            "top_k": args.top_k,
            "alpha": args.alpha,
            "n_neighbors": args.n_neighbors,
            "n_perm": args.mi_n_perm,
            "random_state": args.random_state,
        }
    elif selector_name == "rf_shap":
        selector_kwargs = {
            "mode": args.mode,
            "top_k": args.top_k,
            "alpha": args.alpha,
            "random_state": args.random_state,
            "n_estimators": args.rf_n_estimators,
            "max_depth": args.rf_max_depth,
            "min_samples_leaf": args.rf_min_samples_leaf,
            "val_size": args.rf_val_size,
            "n_repeats": args.rf_n_repeats,
            "pos_weight": args.rf_pos_weight,
            "perm_weight": args.rf_perm_weight,
            "shap_weight": args.rf_shap_weight,
        }
    else:
        raise ValueError(f"Unsupported selector: {selector_name}")

    cluster_df, G = run_method2_sub_selection(
        input_data=args.input_path,
        selector=selector_name,
        label_col=args.label_col,
        protein_prefix=args.protein_prefix,
        n_iter=args.n_iter,
        subsample_size=args.subsample_size,
        random_state=args.random_state,
        selector_kwargs=selector_kwargs,
        pi_thr=args.pi_thr,
        edge_thr=args.edge_thr,
        logger=logger,
    )

    args.cluster_out.parent.mkdir(parents=True, exist_ok=True)
    cluster_df.to_csv(args.cluster_out, index=False)
    logger.info("Saved cluster DataFrame to %s", args.cluster_out)

    args.graph_out.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, args.graph_out)
    logger.info("Saved graph to %s", args.graph_out)

    logger.info("Done in %.2f s", time.time() - start)


if __name__ == "__main__":
    main()
