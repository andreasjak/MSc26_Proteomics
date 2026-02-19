"""
mi_uni_tests.py
---------------
Univariate Mutual Information analysis with permutation-based significance
testing (Benjamini-Hochberg FDR correction) for protein features vs. ARDS label.
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import re
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

import os
from pathlib import Path

import time
start_time = time.time()

# ── Settings ─────────────────────────────────────────────────────────────────
# Set SAVE_RESULTS=1 to disable interactive plotting and only save figures to disk.
SAVE_RESULTS = os.environ.get("SAVE_RESULTS", "0") == "1"

# Use a non-interactive matplotlib backend when saving results to avoid issues in headless environments (e.g. CI, servers).
if SAVE_RESULTS:
    import matplotlib
    matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent.parent # Go up one level to project root
RESULTS_DIR = BASE_DIR / "results"
DATA_PATH = BASE_DIR / "data" / "processed" / "filtered_data.csv"
ANNOT_PATH = BASE_DIR / "data" / "processed" / "somalogic_annotation.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_SIZE       = 0.05
VAL_FRAC        = 0.20          # fraction of remaining data used for validation
RANDOM_STATE    = 42
N_NEIGHBORS     = 3             # k-NN neighbours for MI estimation (Kraskov)
N_PERM          = 1000          # permutations for empirical p-values
FDR_ALPHA       = 0.05

# ── 1. Load data ──────────────────────────────────────────────────────────────
data = pd.read_csv(DATA_PATH, index_col=0, low_memory=False)

X = data.drop(columns=["ards"]).copy()
y = data["ards"].astype(int).copy()

# ── 2. Train / balanced-val / test split ─────────────────────────────────────
row_id = np.arange(len(data))

# 5 % held-out test set (stratified)
X_rem, X_test, y_rem, y_test, id_rem, id_test = train_test_split(
    X, y, row_id,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

rem_df = X_rem.copy()
rem_df["ards"]   = y_rem.values
rem_df["row_id"] = id_rem

df_pos = rem_df[rem_df["ards"] == 1]
df_neg = rem_df[rem_df["ards"] == 0]

n_val_total = int(round(len(rem_df) * VAL_FRAC))
n_each      = min(n_val_total // 2, len(df_pos), len(df_neg))

val_pos = df_pos.sample(n=n_each, random_state=RANDOM_STATE)
val_neg = df_neg.sample(n=n_each, random_state=RANDOM_STATE)
val_df  = pd.concat([val_pos, val_neg]).sample(frac=1, random_state=RANDOM_STATE)

val_ids  = set(val_df["row_id"].values)
train_df = rem_df[~rem_df["row_id"].isin(val_ids)]

X_val   = val_df.drop(columns=["ards", "row_id"])
y_val   = val_df["ards"]
X_train = train_df.drop(columns=["ards", "row_id"])
y_train = train_df["ards"]

# Summary
def _counts(ys):
    return len(ys), int(ys.sum()), int((1 - ys).sum())

for name, ys in [("train", y_train), ("val", y_val), ("test", y_test)]:
    n, n_pos, n_neg = _counts(ys)
    print(f"{name:5s}: {n:4d} patients | ARDS: {n_pos:4d} | non-ARDS: {n_neg:4d}")

print("\nFractions:")
print(f"  test : {len(y_test)/len(y):.3f}")
print(f"  val  : {len(y_val)/len(y):.3f}")
print(f"  train: {len(y_train)/len(y):.3f}")

# ── 3. Univariate Mutual Information ─────────────────────────────────────────
protein_cols = [c for c in X_train.columns if re.match(r"^seq", str(c))]
X_prot = X_train[protein_cols].apply(pd.to_numeric, errors="coerce")
X_prot = X_prot.fillna(X_prot.median())

y_tr = y_train.astype(int).values

mi_scores = mutual_info_classif(
    X_prot.values,
    y_tr,
    discrete_features=False,
    n_neighbors=N_NEIGHBORS,
    random_state=RANDOM_STATE,
)

mi_results = (
    pd.DataFrame({"Protein": protein_cols, "MI": mi_scores})
    .sort_values("MI", ascending=False)
    .reset_index(drop=True)
)

print("\nTop 15 proteins by MI:")
print(mi_results.head(15).to_string(index=False))

# ── 4. Permutation test for significance ─────────────────────────────────────
rng     = np.random.default_rng(RANDOM_STATE)
mi_obs  = mi_results["MI"].values
prot_order = mi_results["Protein"].tolist()
X_perm  = X_prot[prot_order].values

perm_counts = np.zeros(len(prot_order), dtype=int)

print(f"\nRunning {N_PERM} permutations …")
for b in range(N_PERM):
    if N_PERM >= 10 and (b + 1) % (N_PERM // 10) == 0:
        print(f"  Permutation {b + 1}/{N_PERM} ({(b + 1) / N_PERM:.0%})")
        print(f"  Time elapsed: {time.time() - start_time:.2f} seconds")
    y_perm = rng.permutation(y_tr)
    mi_perm = mutual_info_classif(
        X_perm,
        y_perm,
        discrete_features=False,
        n_neighbors=N_NEIGHBORS,
        random_state=RANDOM_STATE,
    )
    perm_counts += (mi_perm >= mi_obs)

p_perm = (perm_counts + 1) / (N_PERM + 1)
mi_results["p_perm"] = p_perm


# Benjamini-Hochberg FDR correction
_, adj_p, _, _ = multipletests(mi_results["p_perm"], method="fdr_bh")
mi_results["ADJ_P"] = adj_p

mi_results = (
    mi_results
    .sort_values(["ADJ_P", "MI"], ascending=[True, False])
    .reset_index(drop=True)
)

print("\nTop 15 proteins after FDR correction:")
print(mi_results.head(15).to_string(index=False))

significant = mi_results[mi_results["ADJ_P"] < FDR_ALPHA]
print(f"\nNumber of significant proteins (ADJ_P < {FDR_ALPHA}): {len(significant)}")

# ── 5. Save results ───────────────────────────────────────────────────────────
if SAVE_RESULTS:
    out_path = RESULTS_DIR / "results_mi_uni_test.csv"
    mi_results.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

print(f"Total runtime: {time.time() - start_time:.2f} seconds")