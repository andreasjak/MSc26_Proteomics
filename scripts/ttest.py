"""
T-test pipeline for proteomics data.

Operates on seen.csv (train + validation set produced by preprocess.py).
Performs:
  1. Data loading
  2. T-tests comparing ARDS vs non-ARDS across all of seen.csv
  3. Multiple testing correction (FDR or Bonferroni)
  4. Volcano plot

Outputs (when --save-results):
  results/<results-subdir>/results_ttest.csv
  results/<results-subdir>/selected_features_k{K}.csv
  results/<results-subdir>/volcano.png
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import logging
import re
import time
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from styles.colors import get_colors

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_LABELS = 20


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(save_results: bool, log_subdir: str, script_name: str) -> logging.Logger:
    """
    Configure logging:
      - save_results=False → StreamHandler (terminal) only
      - save_results=True  → FileHandler (file) only, no terminal output
    Log path: logs/<log_subdir>/<script_name>_YYYYMMDD_HHMMSS.log
    """
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if save_results:
        log_dir = Path("logs") / log_subdir
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"{script_name}_{timestamp}.log"
        handler: logging.Handler = logging.FileHandler(log_path)
    else:
        handler = logging.StreamHandler()

    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def protein_to_probeid(name: str) -> str:
    """Convert protein column name to PROBEID format."""
    s = str(name)
    s = re.sub(r"^seq\.", "", s)
    s = re.sub(r"^seq", "", s)
    s = re.sub(r"([0-9]+)\.([0-9]+)", r"\1-\2", s)
    return s


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def load_data(
    data_path: Path,
    annot_path: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load seen.csv and the SomaLogic annotation table."""
    logger.info("Loading data from %s", data_path)
    data = pd.read_csv(data_path)
    logger.info("Data shape: %s", data.shape)

    try:
        annot = pd.read_csv(annot_path)
        for col in ["PROBEID", "SYMBOL", "UNIPROT", "GENENAME"]:
            if col in annot.columns:
                annot[col] = annot[col].astype(str).replace({"nan": ""})
        logger.info("Annotation loaded: %d rows", len(annot))
    except FileNotFoundError:
        logger.warning("Annotation file not found: %s — proceeding without symbols.", annot_path)
        annot = pd.DataFrame(columns=["PROBEID", "SYMBOL", "UNIPROT", "GENENAME"])

    return data, annot


def run_ttests(data: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Run independent t-tests (Welch) on every seq* protein column,
    comparing ARDS vs non-ARDS groups across the full dataset.
    Returns a DataFrame with raw p-values, mean differences, and group counts.
    """
    X = data.drop(columns=["ards"]).copy()
    y = data["ards"].astype(int).copy()

    n_ards = int((y == 1).sum())
    n_non = int((y == 0).sum())
    logger.info("Running t-tests: %d ARDS / %d non-ARDS samples", n_ards, n_non)

    protein_cols = [c for c in X.columns if re.match(r"^seq", str(c))]
    logger.info("Protein columns to test: %d", len(protein_cols))

    mask1 = y == 1   # ARDS
    mask0 = y == 0   # non-ARDS

    rows = []
    for prot in protein_cols:
        g1 = pd.to_numeric(X.loc[mask1, prot], errors="coerce")
        g0 = pd.to_numeric(X.loc[mask0, prot], errors="coerce")

        g1_clean = g1.dropna()
        g0_clean = g0.dropna()

        if (len(g1_clean) < 2) or (len(g0_clean) < 2):
            continue

        mean_diff = g1_clean.mean() - g0_clean.mean()

        try:
            t = stats.ttest_ind(
                g1_clean.values, g0_clean.values,
                equal_var=False,
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
    logger.info("Proteins tested: %d", len(results))
    return results


def correct_pvalues(
    results: pd.DataFrame,
    method: str,
    alpha: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Apply multiple testing correction and sort by adjusted p-value.
    method: 'fdr' → Benjamini-Hochberg; 'bonferroni' → Bonferroni.
    """
    method_map = {"fdr": "fdr_bh", "bonferroni": "bonferroni"}
    mt_method = method_map[method.lower()]

    _, adj_p, _, _ = multipletests(results["pval"].values, method=mt_method)
    results = results.copy()
    results["ADJ_P"] = adj_p
    results = results.sort_values(["ADJ_P", "pval"], ascending=[True, False]).reset_index(drop=True)

    n_sig = int((results["ADJ_P"] < alpha).sum())
    logger.info(
        "Correction: %s | alpha=%.3f | significant proteins: %d / %d",
        mt_method, alpha, n_sig, len(results),
    )
    logger.info("Top 10 results:\n%s", results.head(10).to_string(index=False))
    return results


def volcano_plot(
    results: pd.DataFrame,
    annot: pd.DataFrame,
    alpha: float,
    n_labels: int,
    save_results: bool,
    results_dir: Path,
    correction_method: str,
    logger: logging.Logger,
) -> None:
    """
    Produce a volcano plot of t-test results.
      - save_results=True  → save to results_dir/volcano.png (no plt.show())
      - save_results=False → plt.show() only (no saving)
    pyplot and adjustText are imported inside this function.
    """
    import matplotlib.pyplot as plt
    from adjustText import adjust_text

    COLOR = get_colors("palette")

    logger.info("Building volcano plot...")

    # Merge annotation
    results = results.copy()
    results["PROBEID"] = results["Protein"].map(protein_to_probeid)
    volcano = results.merge(annot, on="PROBEID", how="left")
    volcano = volcano.drop_duplicates(subset=["Protein"]).reset_index(drop=True)

    volcano["Label"] = np.where(
        volcano["SYMBOL"].notna()
        & (volcano["SYMBOL"].astype(str).str.strip() != "")
        & (volcano["SYMBOL"] != "nan"),
        volcano["SYMBOL"].astype(str),
        volcano["PROBEID"].astype(str),
    )

    # Select top hits for labeling
    sig_mask = volcano["ADJ_P"] < alpha
    top_hits = volcano.loc[sig_mask].copy()
    if len(top_hits) > 0:
        top_hits = top_hits.sort_values("ADJ_P").head(n_labels)
    else:
        top_hits = volcano.sort_values("ADJ_P").head(n_labels)

    top_hits["Label"] = np.where(
        top_hits["SYMBOL"].notna()
        & (top_hits["SYMBOL"].astype(str).str.strip() != "")
        & (top_hits["SYMBOL"] != "nan"),
        top_hits["SYMBOL"].astype(str),
        top_hits["PROBEID"].astype(str),
    )

    x = volcano["MeanDiff"].values
    y = -np.log10(np.clip(volcano["ADJ_P"].values, 1e-300, 1.0))
    is_sig = volcano["ADJ_P"].values < alpha

    plt.figure(figsize=(12, 8))
    plt.scatter(x[~is_sig], y[~is_sig], alpha=0.3, s=20, label="Not significant", color=COLOR["accent"])
    plt.scatter(x[is_sig], y[is_sig], alpha=0.82, s=20, label="Significant", color=COLOR["warning"])

    texts = []
    for _, r in top_hits.iterrows():
        tx = float(r["MeanDiff"])
        ty = -np.log10(max(float(r["ADJ_P"]), 1e-300))
        texts.append(plt.text(tx, ty, str(r["Label"]), fontsize=8, fontweight="bold"))

    adjust_text(texts)

    y_label = "-log10(FDR)" if correction_method.lower() == "fdr" else "-log10(Bonferroni adj. p)"
    plt.xlabel("Mean Difference (ARDS - non-ARDS)")
    plt.ylabel(y_label)
    plt.title(
        f"Volcano Plot: Sepsis with ARDS vs. Sepsis without ARDS\n"
        f"(Top {n_labels} proteins labeled by SYMBOL, fallback to PROBEID)"
    )
    plt.legend(frameon=False)
    plt.tight_layout()

    if save_results:
        volcano_out = results_dir / "volcano.png"
        plt.savefig(volcano_out, dpi=300, bbox_inches="tight")
        logger.info("Saved volcano plot to: %s", volcano_out)
        plt.close()
    else:
        plt.show()

    logger.info(
        "Significant proteins: %d | top %d proteins labeled",
        int(is_sig.sum()), n_labels,
    )
    logger.info(
        "Top 20 most significant:\n%s",
        volcano.sort_values("ADJ_P").head(20)[["PROBEID", "SYMBOL", "MeanDiff", "ADJ_P"]].to_string(index=False),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start = time.time()

    parser = argparse.ArgumentParser(
        description="Run t-tests on seen.csv and produce a volcano plot."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/seen.csv"),
        help="Path to the seen (train+val) CSV (default: data/processed/seen.csv).",
    )
    parser.add_argument(
        "--annot-path",
        type=Path,
        default=Path("data/processed/somalogic_annotation.csv"),
        help="Path to SomaLogic annotation CSV (default: data/processed/somalogic_annotation.csv).",
    )
    parser.add_argument(
        "--results-subdir",
        type=str,
        default="ttest",
        help="Subdirectory under results/ for output files (default: ttest).",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save outputs to disk and log to file; otherwise log to terminal and show plot interactively.",
    )
    parser.add_argument(
        "--log-subdir",
        type=str,
        default="ttest",
        help="Subdirectory under logs/ for log files (default: ttest).",
    )
    parser.add_argument(
        "--correction-method",
        type=str,
        choices=["fdr", "bonferroni"],
        default="fdr",
        help="Multiple testing correction method (default: fdr).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for adjusted p-values (default: 0.05).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of top proteins to save as selected features (default: 10).",
    )
    args = parser.parse_args()

    if args.save_results:
        matplotlib.use("Agg")

    logger = setup_logging(args.save_results, args.log_subdir, "ttest")

    logger.info("Starting ttest.py")
    logger.info(
        "Args: data_path=%s  annot_path=%s  correction=%s  alpha=%s  k=%s  save_results=%s",
        args.data_path, args.annot_path, args.correction_method,
        args.alpha, args.k, args.save_results,
    )

    # Step 1: Load
    data, annot = load_data(args.data_path, args.annot_path, logger)

    # Step 2: T-tests on all of seen.csv
    results = run_ttests(data, logger)

    # Step 3: Multiple testing correction
    results = correct_pvalues(results, args.correction_method, args.alpha, logger)

    # Step 4+5: Volcano plot (always produced; display vs save controlled internally)
    results_dir = Path("results") / args.results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    volcano_plot(
        results, annot, args.alpha, N_LABELS,
        args.save_results, results_dir, args.correction_method,
        logger,
    )

    # Save tabular outputs only when --save-results
    if args.save_results:
        ttest_out = results_dir / "results_ttest.csv"
        results.to_csv(ttest_out, index=False)
        logger.info("Saved t-test results to: %s", ttest_out)

        top_k = results.head(args.k)[["Protein"]].rename(columns={"Protein": "protein"})
        features_out = results_dir / f"selected_features_k{args.k}.csv"
        top_k.to_csv(features_out, index=False)
        logger.info("Saved top-%d features to: %s", args.k, features_out)

    logger.info("Finished in %.2f s", time.time() - start)


if __name__ == "__main__":
    main()
