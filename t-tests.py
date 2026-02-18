"""
T-tests and Classification Pipeline for Proteomics Data
========================================================

This script performs:
1. Data loading and train/val/test splitting
2. T-tests comparing ARDS vs non-ARDS patients
3. Multiple testing correction
4. Volcano plot visualization
5. Feature selection and classification (Logistic Regression, Random Forest, XGBoost)
6. Model evaluation on test set
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from scipy import stats
from statsmodels.stats.multitest import multipletests
from adjustText import adjust_text

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from styles.colors import get_colors


# ============================================================
# Configuration
# ============================================================
DATA_PATH = "data/processed/filtered_data.csv"
ANNOT_PATH = "Attilas/somalogic_annotation.csv"

TEST_SIZE = 0.05
VAL_FRAC = 0.20
RANDOM_STATE = 42

CORRECTION_METHOD = "fdr"  # "fdr" or "bonferroni"
ALPHA_LABEL = 0.05
N_LABELS = 20
K = 10  # Top K proteins for classification

# Colors
ARDS_COLORS = get_colors("ards")
COLOR = get_colors("palette")
COLORS_SEQUENTIAL_TEAL = get_colors("sequential_teal")
COLORS_SEQUENTIAL_ORANGE = get_colors("sequential_orange")


# ============================================================
# Utility Functions
# ============================================================

def protein_to_probeid(name: str) -> str:
    """Convert protein name to PROBEID format."""
    s = str(name)
    s = re.sub(r"^seq\.", "", s)
    s = re.sub(r"^seq", "", s)
    s = re.sub(r"([0-9]+)\.([0-9]+)", r"\1-\2", s)
    return s


def counts(y_):
    """Count total samples, ARDS (1), and non-ARDS (0)."""
    return len(y_), int(y_.sum()), int((1 - y_).sum())


# ============================================================
# Step 1: Load Data
# ============================================================

print("Step 1: Loading data...")
data = pd.read_csv(DATA_PATH, index_col=0)
print(f"Data shape: {data.shape}")
print(data.info())
print()


# ============================================================
# Step 2: Train/Val/Test Split
# ============================================================

print("Step 2: Splitting data into train/val/test...")

# Setup
X = data.drop(columns=["ards"]).copy()
y = data["ards"].astype(int).copy()
row_id = np.arange(len(data))

# 5% test (stratified)
X_rem, X_test, y_rem, y_test, id_rem, id_test = train_test_split(
    X, y, row_id,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# Create balanced validation set
rem_df = X_rem.copy()
rem_df["ards"] = y_rem
rem_df["row_id"] = id_rem

df_pos = rem_df[rem_df["ards"] == 1]
df_neg = rem_df[rem_df["ards"] == 0]

n_val_total = int(round(len(rem_df) * VAL_FRAC))
n_each = min(n_val_total // 2, len(df_pos), len(df_neg))

val_pos = df_pos.sample(n=n_each, random_state=RANDOM_STATE)
val_neg = df_neg.sample(n=n_each, random_state=RANDOM_STATE)

val_df = pd.concat([val_pos, val_neg]).sample(frac=1, random_state=RANDOM_STATE)

# Split using row_id
val_ids = set(val_df["row_id"].values)
train_df = rem_df[~rem_df["row_id"].isin(val_ids)]

X_val = val_df.drop(columns=["ards", "row_id"])
y_val = val_df["ards"]

X_train = train_df.drop(columns=["ards", "row_id"])
y_train = train_df["ards"]

# Print summary
for name, ys in [("train", y_train), ("val", y_val), ("test", y_test)]:
    n, n_pos, n_neg = counts(ys)
    print(f"{name:5s}: {n:4d} patients | ARDS: {n_pos:4d} | non-ARDS: {n_neg:4d}")

print("\nFractions:")
print(f"test: {len(y_test)/len(y):.3f}")
print(f"val : {len(y_val)/len(y):.3f}")
print(f"train: {len(y_train)/len(y):.3f}")
print()


# ============================================================
# Step 3: T-tests (on training data only)
# ============================================================

print("Step 3: Running t-tests...")

protein_cols = [c for c in X_train.columns if re.match(r"^seq", str(c))]

rows = []
mask1 = (y_train == 1)   # ARDS
mask0 = (y_train == 0)   # non-ARDS

for prot in protein_cols:
    g1 = pd.to_numeric(X_train.loc[mask1, prot], errors="coerce")
    g0 = pd.to_numeric(X_train.loc[mask0, prot], errors="coerce")

    # Drop NaNs
    g1_clean = g1.dropna()
    g0_clean = g0.dropna()

    # Need at least 2 values per group
    if (len(g1_clean) < 2) or (len(g0_clean) < 2):
        continue

    mean_diff = g1_clean.mean() - g0_clean.mean()

    try:
        t = stats.ttest_ind(
            g1_clean.values, g0_clean.values,
            equal_var=False
        )
        pval = float(t.pvalue)
    except Exception:
        pval = np.nan

    if np.isfinite(pval):
        rows.append({
            "Protein": prot,
            "MeanDiff": mean_diff,
            "pval": pval,
            "n_ards": len(g1_clean),
            "n_non_ards": len(g0_clean),
        })

results = pd.DataFrame(rows)

# Apply multiple testing correction
method_map = {"fdr": "fdr_bh", "bonferroni": "bonferroni"}
mt_method = method_map.get(CORRECTION_METHOD.lower())
if mt_method is None:
    raise ValueError('CORRECTION_METHOD must be "fdr" or "bonferroni"')

_, adj_p, _, _ = multipletests(results["pval"].values, method=mt_method)
results["ADJ_P"] = adj_p
results = results.sort_values(["ADJ_P", "pval"], ascending=True).reset_index(drop=True)

print(f"Train class counts: {int((y_train==1).sum())} ARDS / {int((y_train==0).sum())} non-ARDS")
print(f"Num proteins tested: {len(results)}")
print(f"Significant proteins (alpha={ALPHA_LABEL}): {sum(results['ADJ_P'] < ALPHA_LABEL)}")
print(results.head(10))
print()


# ============================================================
# Step 4: Prepare annotation and format PROBEID
# ============================================================

print("Step 4: Preparing volcano plot data...")

results["PROBEID"] = results["Protein"].map(protein_to_probeid)

# Load annotation
try:
    annot = pd.read_csv(ANNOT_PATH)
    for col in ["PROBEID", "SYMBOL", "UNIPROT", "GENENAME"]:
        if col in annot.columns:
            annot[col] = annot[col].astype(str).replace({"nan": ""})
except FileNotFoundError:
    print(f"Warning: {ANNOT_PATH} not found")
    annot = pd.DataFrame({"PROBEID": results["PROBEID"].unique()})
    annot["SYMBOL"] = ""
    annot["UNIPROT"] = ""
    annot["GENENAME"] = ""

# Prepare volcano data
volcano = results.merge(annot, on="PROBEID", how="left")
volcano = volcano.drop_duplicates(subset=["Protein"]).reset_index(drop=True)

volcano["Label"] = np.where(
    volcano["SYMBOL"].notna() & (volcano["SYMBOL"].astype(str).str.strip() != "") & (volcano["SYMBOL"] != "nan"),
    volcano["SYMBOL"].astype(str),
    volcano["PROBEID"].astype(str),
)

# Top hits for labeling
sig_mask = volcano["ADJ_P"] < ALPHA_LABEL
top_hits = volcano.loc[sig_mask].copy()
if len(top_hits) > 0:
    top_hits = top_hits.sort_values("ADJ_P").head(N_LABELS)
else:
    top_hits = volcano.sort_values("ADJ_P").head(N_LABELS)

top_hits["Label"] = np.where(
    top_hits["SYMBOL"].notna() & (top_hits["SYMBOL"].astype(str).str.strip() != "") & (top_hits["SYMBOL"] != "nan"),
    top_hits["SYMBOL"].astype(str),
    top_hits["PROBEID"].astype(str),
)

print()


# ============================================================
# Step 5: Volcano Plot
# ============================================================

print("Step 5: Creating volcano plot...")

x = volcano["MeanDiff"].values
y = -np.log10(np.clip(volcano["ADJ_P"].values, 1e-300, 1.0))
is_sig = volcano["ADJ_P"].values < ALPHA_LABEL

plt.figure(figsize=(12, 8))
plt.scatter(x[~is_sig], y[~is_sig], alpha=0.3, s=20, label="Not significant", color=COLOR['accent'])
plt.scatter(x[is_sig], y[is_sig], alpha=0.82, s=20, label="Significant", color=COLOR['warning'])

# Label points
texts = []
for _, r in top_hits.iterrows():
    tx = float(r["MeanDiff"])
    ty = -np.log10(max(float(r["ADJ_P"]), 1e-300))
    texts.append(plt.text(tx, ty, str(r["Label"]), fontsize=8, fontweight="bold"))

adjust_text(texts)

y_label = "-log10(FDR)" if CORRECTION_METHOD.lower() == "fdr" else "-log10(Bonferroni adj. p)"
plt.xlabel("Mean Difference (ARDS - non-ARDS)")
plt.ylabel(y_label)
plt.title(
    f"Volcano Plot: Sepsis with ARDS vs. Sepsis without ARDS\n"
    f"(Top {N_LABELS} proteins labeled by SYMBOL, fallback to PROBEID)"
)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("volcano_plot.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"Significant proteins: {sum(is_sig)}")
print("\nTop 20 most significant:")
print(volcano.sort_values("ADJ_P").head(20)[["PROBEID", "SYMBOL", "MeanDiff", "ADJ_P"]])
print()


# ============================================================
# Step 6: Feature Selection
# ============================================================

print("Step 6: Selecting top proteins for classification...")

topk = results.sort_values("ADJ_P").head(K)["Protein"].tolist()

X_train_k = X_train[topk]
X_val_k = X_val[topk]
X_test_k = X_test[topk]

print(f"X_train_k: {X_train_k.shape}")
print(f"X_val_k: {X_val_k.shape}")
print(f"X_test_k: {X_test_k.shape}")
print(f"y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")
print(y_train.value_counts(), y_val.value_counts(), y_test.value_counts())
print()


# ============================================================
# Step 7: Combine train + val for grid search
# ============================================================

print("Step 7: Combining train + val for grid search...")

X_tv = pd.concat([X_train_k, X_val_k], axis=0)
y_tv = pd.concat([y_train, y_val], axis=0)

test_fold = np.r_[-np.ones(len(X_train_k), dtype=int),
                   np.zeros(len(X_val_k), dtype=int)]
ps = PredefinedSplit(test_fold=test_fold)

print(f"X_tv shape: {X_tv.shape}")
print()


# ============================================================
# Step 8: Logistic Regression
# ============================================================

print("Step 8: Training Logistic Regression...")

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=10000, solver="lbfgs", class_weight="balanced"))
])

param_grid = {"clf__C": [10.0, 15, 18, 21]}

grid = GridSearchCV(
    pipe,
    param_grid=param_grid,
    cv=ps,
    scoring="roc_auc",
    refit=False,
    n_jobs=-1
)

grid.fit(X_tv, y_tv)

print(f"Best params: {grid.best_params_}")
print(f"VAL AUC (grid score): {grid.best_score_:.4f}")

best_model = pipe.set_params(**grid.best_params_)
best_model.fit(X_train_k, y_train)

proba_val = best_model.predict_proba(X_val_k)[:, 1]
pred_val = best_model.predict(X_val_k)

print("\n=== VAL evaluation ===")
print(f"AUC: {roc_auc_score(y_val, proba_val):.4f}")
print(f"Accuracy: {accuracy_score(y_val, pred_val):.4f}")
print(f"Confusion matrix:\n{confusion_matrix(y_val, pred_val)}")
print(classification_report(y_val, pred_val, digits=3))
print()


# ============================================================
# Step 9: Random Forest
# ============================================================

print("Step 9: Training Random Forest...")

pipe_rf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    ))
])

param_grid_rf = {
    "clf__n_estimators": [300, 600],
    "clf__max_depth": [None, 5, 10],
    "clf__min_samples_split": [2, 5, 8, 10],
    "clf__min_samples_leaf": [2, 5],
    "clf__max_features": ["sqrt", 0.5],
}

grid_rf = GridSearchCV(
    pipe_rf,
    param_grid=param_grid_rf,
    cv=ps,
    scoring="roc_auc",
    refit=False,
    n_jobs=-1
)

grid_rf.fit(X_tv, y_tv)

print(f"Best params: {grid_rf.best_params_}")
print(f"VAL AUC (grid score): {grid_rf.best_score_:.4f}")

best_rf = pipe_rf.set_params(**grid_rf.best_params_)
best_rf.fit(X_train_k, y_train)

proba_val = best_rf.predict_proba(X_val_k)[:, 1]
pred_val = best_rf.predict(X_val_k)

print("\n=== VAL evaluation ===")
print(f"AUC: {roc_auc_score(y_val, proba_val):.4f}")
print(f"Accuracy: {accuracy_score(y_val, pred_val):.4f}")
print(f"Confusion matrix:\n{confusion_matrix(y_val, pred_val)}")
print(classification_report(y_val, pred_val, digits=3))
print()


# ============================================================
# Step 10: XGBoost
# ============================================================

print("Step 10: Training XGBoost...")

pipe_xgb = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    ))
])

# Calculate scale_pos_weight
pos = int((y_train == 1).sum())
neg = int((y_train == 0).sum())
spw = (neg / pos) if pos > 0 else 1.0

param_grid_xgb = {
    "clf__n_estimators": [300, 600],
    "clf__learning_rate": [0.03, 0.1],
    "clf__max_depth": [2, 3, 4],
    "clf__subsample": [0.8, 1.0],
    "clf__colsample_bytree": [0.8, 1.0],
    "clf__reg_lambda": [1.0, 10.0],
    "clf__min_child_weight": [1, 3, 5, 10],
    "clf__scale_pos_weight": [1.0, spw],
}

grid_xgb = GridSearchCV(
    pipe_xgb,
    param_grid=param_grid_xgb,
    cv=ps,
    scoring="roc_auc",
    refit=False,
    n_jobs=-1
)

grid_xgb.fit(X_tv, y_tv)

print(f"Best params: {grid_xgb.best_params_}")
print(f"VAL AUC (grid score): {grid_xgb.best_score_:.4f}")

best_xgb = pipe_xgb.set_params(**grid_xgb.best_params_)
best_xgb.fit(X_train_k, y_train)

proba_val = best_xgb.predict_proba(X_val_k)[:, 1]
pred_val = (proba_val >= 0.5).astype(int)

print("\n=== VAL evaluation ===")
print(f"AUC: {roc_auc_score(y_val, proba_val):.4f}")
print(f"Accuracy: {accuracy_score(y_val, pred_val):.4f}")
print(f"Confusion matrix:\n{confusion_matrix(y_val, pred_val)}")
print(classification_report(y_val, pred_val, digits=3))
print()


# ============================================================
# Step 11: Test Set Evaluation
# ============================================================

print("=" * 70)
print("FINAL TEST SET EVALUATION")
print("=" * 70)
print()

# Refit on train+val before final test evaluation
final_model = pipe.set_params(**grid.best_params_)
final_model.fit(X_tv, y_tv)

proba_test = final_model.predict_proba(X_test_k)[:, 1]
pred_test = final_model.predict(X_test_k)

print("Logistic Regression - FINAL TEST evaluation")
print(f"AUC: {roc_auc_score(y_test, proba_test):.4f}")
print(f"Accuracy: {accuracy_score(y_test, pred_test):.4f}")
print(f"Confusion matrix:\n{confusion_matrix(y_test, pred_test)}")
print(classification_report(y_test, pred_test, digits=3))
print()

# Random Forest
final_rf = pipe_rf.set_params(**grid_rf.best_params_)
final_rf.fit(X_tv, y_tv)

proba_test = final_rf.predict_proba(X_test_k)[:, 1]
pred_test = final_rf.predict(X_test_k)

print("Random Forest - FINAL TEST evaluation")
print(f"AUC: {roc_auc_score(y_test, proba_test):.4f}")
print(f"Accuracy: {accuracy_score(y_test, pred_test):.4f}")
print(f"Confusion matrix:\n{confusion_matrix(y_test, pred_test)}")
print(classification_report(y_test, pred_test, digits=3))
print()

# XGBoost
final_xgb = pipe_xgb.set_params(**grid_xgb.best_params_)
final_xgb.fit(X_tv, y_tv)

proba_test = final_xgb.predict_proba(X_test_k)[:, 1]
pred_test = (proba_test >= 0.5).astype(int)

print("XGBoost - FINAL TEST evaluation")
print(f"AUC: {roc_auc_score(y_test, proba_test):.4f}")
print(f"Accuracy: {accuracy_score(y_test, pred_test):.4f}")
print(f"Confusion matrix:\n{confusion_matrix(y_test, pred_test)}")
print(classification_report(y_test, pred_test, digits=3))
print()

print("=" * 70)
print("Pipeline completed successfully!")
print("=" * 70)
