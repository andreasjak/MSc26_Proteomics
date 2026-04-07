"""Methodology 2: subsampling-based feature selection and cluster graph building.

1. Run stratified subsampling selection iterations (t-test, univariate MI, or RF+SHAP selector).
2. Compute selection frequency pi and bootstrap confidence intervals.
3. Build co-selection correlation graph.
4. Detect Louvain communities and return a cluster DataFrame.

Input:
- CSV path OR pandas DataFrame containing protein columns (prefix: seq.) and label column (default: ards)
- Feature selection method name (selector)

Output:
- cluster_df (DataFrame with protein, pi, ci_lower, ci_upper, ci_width, stable_10pct, stable_20pct, cluster)
- G (networkx Graph)
- pi_ci_df (DataFrame with bootstrap CIs for all proteins above pi_thr)
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
        selected = np.zeros(len(adj_p_vals), dtype=int)
        selected[np.argsort(adj_p_vals)[:k]] = 1
    elif mode == "fdr":
        selected = (adj_p_vals < alpha).astype(int)
    else:
        raise ValueError("mode must be 'top_k' or 'fdr'")

    return selected, adj_p_vals


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
    """Return selected mask and adjusted p-values for univariate MI selector."""
    X_num = X_sub.apply(pd.to_numeric, errors="coerce").fillna(X_sub.median())
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
        selected[np.argsort(mi_scores)[::-1][:min(top_k, n_features)]] = 1
        if n_perm >= 1:
            rng = np.random.default_rng(random_state)
            hits = np.zeros(n_features, dtype=int)
            for b in range(n_perm):
                mi_perm = mutual_info_classif(
                    X_num.values,
                    rng.permutation(y_arr),
                    discrete_features=False,
                    n_neighbors=n_neighbors,
                    random_state=random_state + b + 1,
                )
                hits += (mi_perm >= mi_scores).astype(int)
            _, adj_p_vals, _, _ = multipletests(
                (hits + 1) / (n_perm + 1), alpha=alpha, method="fdr_bh"
            )
        else:
            adj_p_vals = np.full(n_features, np.nan)

    elif mode == "fdr":
        if n_perm < 1:
            raise ValueError("uni_mi selector with mode='fdr' requires n_perm >= 1")
        rng = np.random.default_rng(random_state)
        hits = np.zeros(n_features, dtype=int)
        for b in range(n_perm):
            mi_perm = mutual_info_classif(
                X_num.values,
                rng.permutation(y_arr),
                discrete_features=False,
                n_neighbors=n_neighbors,
                random_state=random_state + b + 1,
            )
            hits += (mi_perm >= mi_scores).astype(int)
        _, adj_p_vals, _, _ = multipletests(
            (hits + 1) / (n_perm + 1), alpha=alpha, method="fdr_bh"
        )
        selected = (adj_p_vals < alpha).astype(int)
    else:
        raise ValueError("mode must be 'top_k' or 'fdr'")

    return selected, adj_p_vals


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
    """Select proteins using RF permutation importance + SHAP class-1 importance."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "rf_shap selector requires the 'shap' package."
        ) from exc

    X_num = X_sub.apply(pd.to_numeric, errors="coerce").fillna(X_sub.median())
    y_arr = y_sub.astype(int).values

    X_train, X_val, y_train, y_val = train_test_split(
        X_num, y_arr, test_size=val_size, stratify=y_arr, random_state=random_state
    )

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight={0: 1.0, 1: float(pos_weight)},
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    perm = permutation_importance(
        rf, X_val, y_val,
        scoring=make_scorer(recall_score, pos_label=1),
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    perm_mean = np.clip(perm.importances_mean, a_min=0.0, a_max=None)
    perm_p = ((perm.importances <= 0.0).sum(axis=1) + 1) / (n_repeats + 1)

    shap_vals = shap.TreeExplainer(rf).shap_values(X_val)
    if isinstance(shap_vals, list):
        shap_arr = np.asarray(shap_vals[1] if len(shap_vals) > 1 else shap_vals[0])
    else:
        shap_arr = np.asarray(shap_vals)
        if shap_arr.ndim == 3:
            shap_arr = shap_arr[:, :, 1] if shap_arr.shape[2] > 1 else shap_arr[:, :, 0]
    shap_mean_abs = np.mean(np.abs(shap_arr), axis=0)

    def _norm(v: np.ndarray) -> np.ndarray:
        s = float(np.asarray(v, dtype=float).sum())
        return np.zeros_like(v) if s <= 0 else np.asarray(v, dtype=float) / s

    w_sum = max(perm_weight, 0.0) + max(shap_weight, 0.0)
    w_perm = max(perm_weight, 0.0) / w_sum if w_sum > 0 else 0.5
    w_shap = max(shap_weight, 0.0) / w_sum if w_sum > 0 else 0.5
    combined_score = w_perm * _norm(perm_mean) + w_shap * _norm(shap_mean_abs)

    _, adj_p_vals, _, _ = multipletests(perm_p, alpha=alpha, method="fdr_bh")
    n_features = X_num.shape[1]
    selected = np.zeros(n_features, dtype=int)

    if mode == "top_k":
        selected[np.argsort(combined_score)[::-1][:min(top_k, n_features)]] = 1
    elif mode == "fdr":
        selected = (adj_p_vals < alpha).astype(int)
    else:
        raise ValueError("mode must be 'top_k' or 'fdr'")

    return selected, adj_p_vals


SELECTOR_REGISTRY: dict[str, Callable[..., tuple[np.ndarray, np.ndarray]]] = {
    "ttest": run_ttest_selector,
    "uni_mi": uni_mi_selector,
    "rf_shap": rf_shap_selector,
}

_ALIAS_MAP = {
    "ttest": "ttest", "t_test": "ttest",
    "uni_mi": "uni_mi", "mi_uni": "uni_mi", "mi": "uni_mi",
    "rf_shap": "rf_shap", "rd_shap": "rf_shap",
}


def _normalize_selector(selector: str) -> str:
    key = str(selector).strip().lower()
    if key not in _ALIAS_MAP:
        raise ValueError(f"Unknown selector: {selector}. Available: {list(SELECTOR_REGISTRY)}")
    return _ALIAS_MAP[key]


# ---------------------------------------------------------------------------
# Core workflow
# ---------------------------------------------------------------------------


def _load_input_data(input_data: pd.DataFrame | str | Path) -> pd.DataFrame:
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
    """Run stratified subsampling and build Z (selection matrix) and P (adj p-values)."""
    selector = _normalize_selector(selector)
    selector_fn = SELECTOR_REGISTRY[selector]
    selector_kwargs = selector_kwargs or {}

    n_samples, n_proteins = X.shape
    Z = np.zeros((n_iter, n_proteins), dtype=int)
    P = np.zeros((n_iter, n_proteins), dtype=float)

    for i in range(n_iter):
        sub_idx, _ = train_test_split(
            np.arange(n_samples),
            train_size=subsample_size,
            stratify=y,
            random_state=random_state + i,
        )
        selected, adj_p_vals = selector_fn(X.iloc[sub_idx], y.iloc[sub_idx], **selector_kwargs)
        Z[i] = selected
        P[i] = adj_p_vals

        if logger is not None and (i + 1) % 100 == 0:
            logger.info("Iteration %d/%d", i + 1, n_iter)

    return Z, P


def bootstrap_pi_ci(
    Z: np.ndarray,
    proteins: list[str],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute bootstrapped confidence intervals for selection frequency pi_i.

    Resamples rows of Z (iterations) with replacement, making no independence
    assumption. This naturally captures the dependency structure introduced by
    overlapping subsamples.

    Parameters
    ----------
    Z           : (n_iter x n_proteins) binary selection matrix
    proteins    : protein names matching columns of Z
    n_bootstrap : number of bootstrap resamples
    alpha       : significance level (0.05 -> 95% CI)
    random_state: random seed

    Returns
    -------
    DataFrame with columns: protein, pi, ci_lower, ci_upper, ci_width,
                            stable_10pct, stable_20pct
    """
    rng = np.random.default_rng(random_state)
    n_iter = Z.shape[0]

    # Shape: (n_bootstrap, n_proteins) — pi estimate per resample
    pi_boots = np.array([
        Z[rng.integers(0, n_iter, size=n_iter)].mean(axis=0)
        for _ in range(n_bootstrap)
    ])

    return pd.DataFrame({
        "protein": proteins,
        "pi": Z.mean(axis=0),
        "ci_lower": np.percentile(pi_boots, 100 * alpha / 2, axis=0),
        "ci_upper": np.percentile(pi_boots, 100 * (1 - alpha / 2), axis=0),
        "ci_width": np.percentile(pi_boots, 100 * (1 - alpha / 2), axis=0)
                  - np.percentile(pi_boots, 100 * alpha / 2, axis=0),
        "stable_10pct": np.percentile(pi_boots, 100 * alpha / 2, axis=0) > 0.10,
        "stable_20pct": np.percentile(pi_boots, 100 * alpha / 2, axis=0) > 0.20,
    }).sort_values("pi", ascending=False).reset_index(drop=True)


def build_graph_and_clusters(
    Z: np.ndarray,
    proteins: list[str],
    pi_thr: float = 0.05,
    edge_thr: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, nx.Graph]:
    """Filter proteins by pi, build co-selection correlation graph, detect communities."""
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

    R = np.nan_to_num(np.corrcoef(Z_filt, rowvar=False), nan=0.0)
    p = len(proteins_filt)
    for i in range(p):
        for j in range(i + 1, p):
            if R[i, j] > edge_thr:
                G.add_edge(str(proteins_filt[i]), str(proteins_filt[j]), weight=float(R[i, j]))

    if G.number_of_edges() == 0:
        communities: list[set[str]] = [{n} for n in G.nodes()]
    else:
        communities = nx.community.louvain_communities(G, weight="weight", seed=random_state)

    cluster_map: dict[str, int] = {}
    for k, comm in enumerate(communities):
        for prot in comm:
            cluster_map[prot] = k
    for prot in proteins_filt:
        cluster_map.setdefault(str(prot), -1)

    return pd.DataFrame({
        "protein": proteins_filt.astype(str),
        "pi": pi_filt,
        "cluster": [cluster_map[str(p)] for p in proteins_filt],
    }).sort_values(["cluster", "pi"], ascending=[True, False]).reset_index(drop=True), G


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
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, nx.Graph, pd.DataFrame]:
    """End-to-end Methodology 2 pipeline.

    Returns
    -------
    cluster_df : proteins with pi, bootstrap CIs, and Louvain cluster assignment
    G          : co-selection graph
    pi_ci_df   : bootstrap CIs for all proteins (unfiltered)
    """
    selector = _normalize_selector(selector)
    df = _load_input_data(input_data)

    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    proteins = [c for c in df.columns if str(c).startswith(protein_prefix)]
    if not proteins:
        raise ValueError(f"No protein columns found with prefix '{protein_prefix}'.")

    X = df[proteins].copy()
    y = df[label_col].astype(int).copy()

    if logger is not None:
        logger.info("Input shape: X=%s, y=%s", X.shape, y.shape)
        logger.info("Selector: %s", selector)

    Z, _ = run_subsampling_selection(
        X=X, y=y, selector=selector, n_iter=n_iter,
        subsample_size=subsample_size, random_state=random_state,
        selector_kwargs=selector_kwargs, logger=logger,
    )

    # Bootstrap CIs on raw Z — before pi_thr filtering so boundary proteins
    # also get honest uncertainty estimates
    if logger is not None:
        logger.info("Computing bootstrap CIs (n_bootstrap=%d)...", n_bootstrap)

    pi_ci_df = bootstrap_pi_ci(
        Z=Z, proteins=proteins,
        n_bootstrap=n_bootstrap, alpha=alpha, random_state=random_state,
    )

    if logger is not None:
        logger.info("Proteins with CI lower bound > 10%%: %d", pi_ci_df["stable_10pct"].sum())

    cluster_df, G = build_graph_and_clusters(
        Z=Z, proteins=proteins, pi_thr=pi_thr, edge_thr=edge_thr, random_state=random_state,
    )

    # Merge CI columns into cluster_df
    cluster_df = cluster_df.merge(
        pi_ci_df[["protein", "ci_lower", "ci_upper", "ci_width", "stable_10pct", "stable_20pct"]],
        on="protein", how="left",
    )

    if logger is not None:
        logger.info(
            "Done: %d proteins, %d graph nodes, %d graph edges",
            len(cluster_df), G.number_of_nodes(), G.number_of_edges(),
        )

    return cluster_df, G, pi_ci_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    start = time.time()

    parser = argparse.ArgumentParser(
        description="Methodology 2: subsampling selection -> bootstrap CIs -> graph -> Louvain clusters."
    )
    parser.add_argument("--input-path", type=Path, default=Path("data/processed/seen.csv"))
    parser.add_argument("--selector", type=str, default="ttest",
                        help="Selector: ttest | uni_mi | rf_shap")
    parser.add_argument("--method", type=str, default=None,
                        help="Alias for --selector. Overrides --selector if set.")
    parser.add_argument("--mode", type=str, default="top_k", choices=["top_k", "fdr"])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-neighbors", type=int, default=3,
                        help="k for mutual_info_classif (uni_mi only)")
    parser.add_argument("--mi-n-perm", type=int, default=0,
                        help="Permutations for MI p-values (uni_mi only)")
    parser.add_argument("--rf-n-estimators", type=int, default=300)
    parser.add_argument("--rf-max-depth", type=int, default=None)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1)
    parser.add_argument("--rf-val-size", type=float, default=0.3)
    parser.add_argument("--rf-n-repeats", type=int, default=20)
    parser.add_argument("--rf-pos-weight", type=float, default=3.0)
    parser.add_argument("--rf-perm-weight", type=float, default=0.6)
    parser.add_argument("--rf-shap-weight", type=float, default=0.4)
    parser.add_argument("--n-iter", type=int, default=300)
    parser.add_argument("--subsample-size", type=float, default=0.5)
    parser.add_argument("--pi-thr", type=float, default=0.05)
    parser.add_argument("--edge-thr", type=float, default=0.15)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--label-col", type=str, default="ards")
    parser.add_argument("--protein-prefix", type=str, default="seq.")
    parser.add_argument("--cluster-out", type=Path,
                        default=Path("results/method2_sub_selection/cluster_df.csv"))
    parser.add_argument("--ci-out", type=Path,
                        default=Path("results/method2_sub_selection/pi_ci_df.csv"))
    parser.add_argument("--graph-out", type=Path,
                        default=Path("results/method2_sub_selection/graph.graphml"))
    parser.add_argument("--save-results", action="store_true")
    args = parser.parse_args()

    logger = setup_logging(
        save_results=args.save_results,
        log_subdir="method2_sub_selection",
        script_name="method2_sub_selection",
    )

    selector_name = _normalize_selector(args.method if args.method is not None else args.selector)

    selector_kwargs: dict[str, Any]
    if selector_name == "ttest":
        selector_kwargs = {"mode": args.mode, "top_k": args.top_k, "alpha": args.alpha}
    elif selector_name == "uni_mi":
        selector_kwargs = {
            "mode": args.mode, "top_k": args.top_k, "alpha": args.alpha,
            "n_neighbors": args.n_neighbors, "n_perm": args.mi_n_perm,
            "random_state": args.random_state,
        }
    elif selector_name == "rf_shap":
        selector_kwargs = {
            "mode": args.mode, "top_k": args.top_k, "alpha": args.alpha,
            "random_state": args.random_state, "n_estimators": args.rf_n_estimators,
            "max_depth": args.rf_max_depth, "min_samples_leaf": args.rf_min_samples_leaf,
            "val_size": args.rf_val_size, "n_repeats": args.rf_n_repeats,
            "pos_weight": args.rf_pos_weight, "perm_weight": args.rf_perm_weight,
            "shap_weight": args.rf_shap_weight,
        }
    else:
        raise ValueError(f"Unsupported selector: {selector_name}")

    cluster_df, G, pi_ci_df = run_method2_sub_selection(
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
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha,
        logger=logger,
    )

    for path, data in [
        (args.cluster_out, cluster_df),
        (args.ci_out, pi_ci_df),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)
        logger.info("Saved %s", path)

    args.graph_out.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, args.graph_out)
    logger.info("Saved %s", args.graph_out)
    logger.info("Done in %.2f s", time.time() - start)


if __name__ == "__main__":
    main()
