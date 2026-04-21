"""Global configuration constants for the ARDS proteomics pipeline.

This module is the single source of truth for shared parameters used across
pipeline stages.
"""

from __future__ import annotations

from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"

# Resampling
K_SPLITS = 50
K_SPLITS_EXPENSIVE = 10
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Top-k values for evaluation
TOPK_VALUES = [10, 25, 50, 100]

# Stability
STABILITY_PI_PRIMARY = 0.2
STABILITY_MIN_SIZE = 10
STABILITY_TOPK_FALLBACK = 20

# Enrichment
BES_CAP_C = 10
BES_JACCARD_TAU = 0.5
ENRICHMENT_Q_THRESHOLD = 0.05
ENRICHMENT_LIBRARIES = [
	"GO_Biological_Process_2025",
	"KEGG_2026",
	"Reactome_Pathways_2024",
]
PERMUTATION_B = 1000

# Classifiers (fixed methodology settings)
LR_CONFIG = {
	"class_weight": "balanced",
	"max_iter": 1000,
}

RF_CONFIG = {
	"class_weight": "balanced",
	"n_estimators": 100,
	"max_depth": 5,
	"random_state": RANDOM_SEED,
	"n_jobs": -1,
}

XGB_CONFIG = {
	"n_estimators": 100,
	"max_depth": 5,
	"eval_metric": "auc",
	"random_state": RANDOM_SEED,
}

# XGBoost class-imbalance policy: compute scale_pos_weight as n_neg / n_pos
# per training fold.
XGB_SCALE_POS_WEIGHT_MODE = "n_neg_over_n_pos"

# Simulation
SIM_N = 400
SIM_P = 100
SIM_CLASS_PREVALENCE = 0.2
SIM_SIGNALS_PER_TYPE = 3
SIM_EFFECT_SIZES = [0.3, 0.6, 1.0]
SIM_REPEATS = 50


__all__ = [
	"REPO_ROOT",
	"DATA_RAW",
	"DATA_PROCESSED",
	"RESULTS_DIR",
	"K_SPLITS",
	"K_SPLITS_EXPENSIVE",
	"TEST_SIZE",
	"RANDOM_SEED",
	"TOPK_VALUES",
	"STABILITY_PI_PRIMARY",
	"STABILITY_MIN_SIZE",
	"STABILITY_TOPK_FALLBACK",
	"BES_CAP_C",
	"BES_JACCARD_TAU",
	"ENRICHMENT_Q_THRESHOLD",
	"ENRICHMENT_LIBRARIES",
	"PERMUTATION_B",
	"LR_CONFIG",
	"RF_CONFIG",
	"XGB_CONFIG",
	"XGB_SCALE_POS_WEIGHT_MODE",
	"SIM_N",
	"SIM_P",
	"SIM_CLASS_PREVALENCE",
	"SIM_SIGNALS_PER_TYPE",
	"SIM_EFFECT_SIZES",
	"SIM_REPEATS",
]
