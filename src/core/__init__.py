"""
src.core
--------
Core reusable modules for the MSc26 proteomics project.

Modules
-------
data_utils
    Dataset splitting, protein-feature extraction, and top-differential-feature
    identification.
plot_utils
    Correlation matrix computation and hierarchically-clustered heatmap plotting.
"""

from .data_utils import create_subsets, get_protein_features, get_top_diff_features
from .plot_utils import corr_matrix, hierarchical_feature_order, plot_correlation_heatmap

__all__ = [
    "create_subsets",
    "get_protein_features",
    "get_top_diff_features",
    "corr_matrix",
    "hierarchical_feature_order",
    "plot_correlation_heatmap",
]