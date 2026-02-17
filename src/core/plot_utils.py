"""
plot_utils.py
-------------
Reusable plotting utilities for the MSc26 proteomics project.

Provides correlation matrix computation and hierarchically-clustered heatmap
plotting used in EDA and downstream analyses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------

def corr_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Compute a Pearson correlation matrix for *features* in *df*.

    Missing values are imputed column-wise with the column median before
    computing correlations, so the result is always finite.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the feature columns.
    features : list[str]
        Column names to include in the correlation matrix.

    Returns
    -------
    pd.DataFrame
        Square correlation matrix with shape ``(len(features), len(features))``.
    """
    mat = df[features].copy().astype(float)
    mat = mat.apply(lambda col: col.fillna(col.median()), axis=0)
    return mat.corr()


def hierarchical_feature_order(corr_df: pd.DataFrame) -> list[str]:
    """Return feature names re-ordered by hierarchical clustering (average linkage).

    Uses correlation distance ``1 - r`` as the dissimilarity measure, which
    groups co-expressed proteins together in heatmaps.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Square correlation matrix (as returned by :func:`corr_matrix`).

    Returns
    -------
    list[str]
        Feature names in the order determined by the dendrogram leaf order.
    """
    C = np.nan_to_num(corr_df.to_numpy(), nan=0.0)
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    order = dendrogram(Z, no_plot=True)["leaves"]
    features = list(corr_df.columns)
    return [features[i] for i in order]


def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    title: str,
    ordered_features: list[str] | None = None,
    ax: plt.Axes | None = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "RdBu_r",
) -> plt.Axes:
    """Plot a correlation heatmap with optional hierarchical feature ordering.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Square correlation matrix.
    title : str
        Plot title.
    ordered_features : list[str] or None, optional
        Pre-computed leaf order from :func:`hierarchical_feature_order`.
        If *None*, the original column order is used.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on.  A new figure is created when *None*.
    vmin, vmax : float, optional
        Colour scale limits.  Defaults to ``[-1, 1]``.
    cmap : str, optional
        Matplotlib colormap name.  Defaults to ``"RdBu_r"``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the heatmap.
    """
    if ordered_features is not None:
        M = corr_df.loc[ordered_features, ordered_features].to_numpy()
    else:
        M = corr_df.to_numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.get_figure()

    im = ax.imshow(M, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(im, ax=ax, label="Correlation")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax