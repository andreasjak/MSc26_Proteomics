"""
src.core
--------
Core reusable modules for the MSc26 proteomics project.

Modules
-------
classifier_utils
    Monte Carlo CV splitter, pipeline factories (LR, RF, XGBoost), and
    generic train / evaluate functions for classification pipelines.
    Import directly: ``from src.core.classifier_utils import ...``
    (not auto-imported here to avoid mandatory xgboost/sklearn dependency).
data_utils
    Dataset splitting, protein-feature extraction, top-differential-feature
    identification, and generic data / annotation / feature-list loading.
logging_utils
    Shared logging configuration (terminal or file-based).
plot_utils
    Correlation matrix computation and hierarchically-clustered heatmap plotting.
"""

from .data_utils import (
    create_subsets,
    filter_data,
    get_protein_features,
    get_top_diff_features,
    load_annotation,
    load_data,
    load_features,
)
from .logging_utils import setup_logging
from .plot_utils import corr_matrix, hierarchical_feature_order, plot_correlation_heatmap

__all__ = [
    "create_subsets",
    "filter_data",
    "get_protein_features",
    "get_top_diff_features",
    "load_annotation",
    "load_data",
    "load_features",
    "setup_logging",
    "corr_matrix",
    "hierarchical_feature_order",
    "plot_correlation_heatmap",
]