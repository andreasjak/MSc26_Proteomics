
# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.multitest import multipletests

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from styles.colors import get_colors
from src.core.data_utils import load_annotation, load_data
from src.core.logging_utils import setup_logging

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOG_SUBDIR = "ttest_pipeline"
SAVE_RESULTS = False
N_ITER = 100

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def val_split(X, y, val_size=0.2, random_state=42):
    """Create a balanced validation split from the seen data."""
    rng = np.random.default_rng(random_state)

    pos_idx = y[y == 1].index.to_numpy()
    neg_idx = y[y == 0].index.to_numpy()

    n_val_each = int(len(y) * val_size) // 2
    if n_val_each < 1:
        raise ValueError("Not enough samples to build a balanced validation set.")
    
    if len(pos_idx) < n_val_each or len(neg_idx) < n_val_each:
        raise ValueError("Not enough samples in one of the classes to build a balanced validation set.")

    val_pos = rng.choice(pos_idx, size=n_val_each, replace=False)
    val_neg = rng.choice(neg_idx, size=n_val_each, replace=False)
    val_idx = np.concatenate([val_pos, val_neg])

    train_idx = y.index.difference(val_idx)

    return X.loc[train_idx], y.loc[train_idx], X.loc[val_idx], y.loc[val_idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ## Setup
    logger = setup_logging(SAVE_RESULTS, LOG_SUBDIR, "pipeline")

    ## Load filtered data
    df = load_data("data/processed/filtered_data.csv", logger)
    #print(f"Loaded data with shape: {df.shape}")

    proteins = [c for c in df.columns if c.startswith("seq.")]
    X = df[proteins]
    y = df["ards"]

    ## Split in seen and test sets
    X_seen, X_test, y_seen, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    ## Define classifier 
    rf_clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, max_depth=6, n_jobs=-1)

    # Keep track of which proteins are selected in each iteration
    proteins_selected = {prot: [] for prot in proteins}
    
    # Keep track of model performance across iterations
    performance_metrics = {
        "accuracy": [],
        "f1_score": [],
        "roc_auc": []
    }

    for i in range(N_ITER):
        X_train, y_train, X_val, y_val = val_split(X_seen, y_seen, val_size=0.2, random_state=42+i)

        ## Select proteins using t-tests across bootstrap iterations, and evaluate the stability of selection across iterations
        p_threshold = 0.05
        # Dictionary to store results: Key = Protein Name, Value = Dict with lists
        protein_stats = {prot: {"p_values": [], "is_significant": []} for prot in proteins}
       
        # Perform t-test (vectorized operation for all proteins)
        t_stats, p_vals = stats.ttest_ind(
            X_train[y_train == 1], 
            X_train[y_train == 0], 
            axis=0, 
            equal_var=False, 
            nan_policy='omit'
        )

        # Multiple testing correction
        reject, p_vals_corrected, _, _ = multipletests(p_vals, alpha=p_threshold, method='fdr_bh')

        # Selected proteins based on corrected p-values
        sig_proteins = [prot for prot, is_sig in zip(proteins, reject) if is_sig]

        # Update the list of selected proteins for each iteration
        for prot in proteins:
            proteins_selected[prot].append(prot in sig_proteins)

        logger.info(f"Number of proteins selected in iteration {i+1}: {len(sig_proteins)}")

        # Train the classifier on the selected proteins
        if len(sig_proteins) > 0:
            rf_clf.fit(X_train[sig_proteins], y_train)
            y_pred = rf_clf.predict(X_val[sig_proteins])
            y_prob = rf_clf.predict_proba(X_val[sig_proteins])[:, 1]

            # Evaluate performance
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            roc_auc = roc_auc_score(y_val, y_prob)

            performance_metrics["accuracy"].append(acc)
            performance_metrics["f1_score"].append(f1)
            performance_metrics["roc_auc"].append(roc_auc)

            logger.info(f"Iteration {i+1} - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
        else:
            logger.info(f"Iteration {i+1} - No significant proteins selected. Skipping model training.")


    # After all iterations, analyze the stability of protein selection and model performance
    selection_stability = {prot: np.mean(selected) for prot, selected in proteins_selected.items()}
    logger.info("Feature selection stability across iterations:")
    for prot, stability in selection_stability.items():
        if stability > 0.2:  # Only log proteins that were selected in more than 50% of iterations
            logger.info(f"{prot}: {stability:.2f}")

    # print number of proteins with stability > 0.2
    stable_proteins = [prot for prot, stability in selection_stability.items() if stability > 0.2]
    logger.info(f"Number of proteins with selection stability > 0.2: {len(stable_proteins)}")

    # Analyze performance metrics across iterations (mean + standard deviation)
    for metric, values in performance_metrics.items():
        logger.info(f"{metric.capitalize()}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    rf_clf.fit(X_seen[stable_proteins], y_seen)
    y_test_pred = rf_clf.predict(X_test[stable_proteins])
    y_test_prob = rf_clf.predict_proba(X_test[stable_proteins])

    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_prob[:, 1])

    logger.info(f"Test Set Performance - Accuracy: {test_acc:.4f}, F1 Score: {test_f1:.4f}, ROC AUC: {test_roc_auc:.4f}")

    # confusion matrix with numbers
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    # Add numbers to confusion matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=12)

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["No ARDS", "ARDS"], rotation=45)
    plt.yticks(tick_marks, ["No ARDS", "ARDS"])
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    # Plot AUC curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_test_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()