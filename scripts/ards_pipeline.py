"""
ARDS Enrichment Pipeline — Three-Stage Analysis
================================================
Inputs (set at the top of the script):
    selected_proteins_df : DataFrame with columns ["proteins", "importance"]
                           (RF feature importances across iterations)
    all_data_df          : All patients, rows=patients, columns=seq_ids + clinical
    seen_df              : Training patients, same format as all_data_df
    unseen_df            : Test patients, same format as all_data_df
    anno                 : SomaScan annotation DataFrame
    rename_proteins_to_symbol : your existing mapping function
    make_enrichr_gene_list    : your existing gene list function

Usage:
    Run from your notebook:
        %run ards_pipeline.py
    Or import:
        from ards_pipeline import run_pipeline
        results = run_pipeline(selected_proteins_df, all_data_df, seen_df, unseen_df, anno)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import gseapy as gp

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, roc_auc_score, f1_score,
    balanced_accuracy_score, RocCurveDisplay, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import pdist, squareform
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these
# ══════════════════════════════════════════════════════════════════════════════
IMPORTANCE_CUTOFF = 0.0001   # min RF importance to keep a protein
FREQ_CUTOFF       = 0.05     # if selected_proteins_df has frequency instead
Z_THRESHOLD       = 0.0      # z-score threshold for per-patient hit lists
FDR_CUTOFF        = 0.05     # enrichment significance threshold
MIN_PATIENTS_PCT  = 0.20     # pathway must be significant in this fraction of ARDS patients
LABEL_COL         = "ards"   # column name for ARDS label in dataframes
SAMPLE_ID_COL     = "SampleId"  # patient ID column

GENE_SETS = [
    "Reactome_2022",
    "GO_Biological_Process_2023",
    "KEGG_2021_Human",
    "MSigDB_Hallmark_2020",
]

# Colours
C_ARDS = "#c0392b"
C_CTRL = "#2980b9"
ENDOTYPE_PALETTE = {0: "#E74C3C", 1: "#3498DB", 2: "#2ECC71", 3: "#F39C12"}
SEV_PALETTE = {"Mild": "#F9E79F", "Moderate": "#F39C12", "Severe": "#c0392b"}

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def prepare_patient_df(df, label_col, sample_id_col):
    """Reset index to SampleId, split labels from proteins."""
    df = df.copy()
    if sample_id_col in df.columns:
        df = df.set_index(sample_id_col)
    elif df.index.name != sample_id_col:
        df = df.reset_index().set_index(sample_id_col)
    protein_cols = [c for c in df.columns if c.startswith("seq.")]
    labels       = df[label_col].copy()
    proteins     = df[protein_cols].copy()
    return proteins, labels


def build_seq_to_gene(anno):
    """Build seq.XXXXX → gene symbol mapping from SomaScan annotation."""
    anno = anno.copy()
    anno["seq_id"] = "seq." + anno["PROBEID"].str.replace("-", ".")
    return (anno.dropna(subset=["SYMBOL"])
                .set_index("seq_id")["SYMBOL"]
                .to_dict())


def rename_df_columns(df, seq_to_gene, duplicate_strategy="first"):
    """Rename seq columns to gene names, drop unmapped."""
    rename_map = {c: seq_to_gene[c] for c in df.columns if c in seq_to_gene}
    renamed    = df.rename(columns=rename_map)
    # Keep only renamed columns
    kept = [c for c in renamed.columns if c in rename_map.values()]
    renamed = renamed[kept]
    # Handle duplicates
    if duplicate_strategy == "first":
        renamed = renamed.loc[:, ~renamed.columns.duplicated(keep="first")]
    return renamed


def pathway_score_fn(adj_p, cutoff=FDR_CUTOFF):
    if adj_p >= cutoff:
        return 0.0
    return -np.log10(max(adj_p, 1e-300))


def make_pipeline(C=0.1, n_estimators=1000):
    return {
        "LR": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", C=C,
                max_iter=2000, solver="saga", random_state=42))
        ]),
        "RF": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=n_estimators, class_weight="balanced",
                max_features="sqrt", max_depth=6,
                min_samples_leaf=3, n_jobs=-1, random_state=42))
        ]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def stage1_validation(robust_genes, background, gene_sets=GENE_SETS):
    print("\n" + "="*65)
    print("STAGE 1 — Validation: enrichment on RF-selected proteins")
    print("="*65)

    enr = gp.enrichr(
        gene_list=robust_genes,
        gene_sets=gene_sets,
        background=background,
        outdir=None,
        cutoff=1.0,
    )
    results = enr.results.copy()
    sig     = results[results["Adjusted P-value"] < FDR_CUTOFF].copy()
    sig     = sig.sort_values("Adjusted P-value")

    print(f"Significant pathways (adj p < {FDR_CUTOFF}): {len(sig)}")
    if len(sig) > 0:
        print(sig[["Term", "Adjusted P-value", "Genes"]].head(10).to_string())

    # ── Plot: enrichment dot plot ─────────────────────────────────────────────
    if len(sig) > 0:
        top20 = sig.head(20).copy()
        top20["-log10p"] = -np.log10(top20["Adjusted P-value"])
        top20["n_genes"] = top20["Genes"].apply(
            lambda x: len(x.split(";")) if pd.notna(x) else 0)
        top20 = top20.sort_values("-log10p")

        fig, ax = plt.subplots(figsize=(10, 7))
        sc = ax.scatter(
            top20["-log10p"], range(len(top20)),
            s=top20["n_genes"] * 25,
            c=top20["-log10p"], cmap="YlOrRd",
            alpha=0.85, edgecolors="grey", linewidth=0.4
        )
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(top20["Term"].str[:55], fontsize=7)
        ax.set_xlabel("-log10(adjusted p-value)")
        ax.set_title("Stage 1 — Top enriched pathways in RF-selected proteins\n"
                     "(dot size = number of genes)", fontweight="bold")
        plt.colorbar(sc, ax=ax, label="-log10(adj p)")
        plt.tight_layout()
        plt.savefig("stage1_enrichment.png", dpi=150, bbox_inches="tight")
        plt.show()

    # ── Plot: mean z-score difference ─────────────────────────────────────────
    return sig, results


def stage1_plots(pat_renamed, y_train, sig_validation):
    """Protein-level plots for Stage 1."""
    control_mean = pat_renamed.loc[y_train==0].mean()
    control_std  = pat_renamed.loc[y_train==0].std().replace(0, 1)
    zscored_all  = (pat_renamed - control_mean) / control_std

    mean_z = pd.DataFrame({
        "Control": zscored_all.loc[y_train==0].mean(),
        "ARDS"   : zscored_all.loc[y_train==1].mean(),
    })
    mean_z["Difference"] = mean_z["ARDS"] - mean_z["Control"]
    mean_z = mean_z.sort_values("Difference", ascending=False)

    # Mean difference bar
    fig, ax = plt.subplots(figsize=(16, 5))
    bar_colors = [C_ARDS if v > 0 else C_CTRL for v in mean_z["Difference"]]
    ax.bar(range(len(mean_z)), mean_z["Difference"],
           color=bar_colors, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(mean_z)))
    ax.set_xticklabels(mean_z.index, rotation=90, fontsize=6)
    ax.set_ylabel("Mean z-score difference (ARDS − Control)")
    ax.set_title("Stage 1 — All RF proteins: elevation in ARDS vs Control",
                 fontweight="bold")
    ax.legend(handles=[Patch(color=C_ARDS, label="Higher in ARDS"),
                       Patch(color=C_CTRL, label="Higher in Control")])
    plt.tight_layout()
    plt.savefig("stage1_mean_diff.png", dpi=150, bbox_inches="tight")
    plt.show()

    # PCA
    pca   = PCA(n_components=2)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(pat_renamed))

    fig, ax = plt.subplots(figsize=(7, 6))
    for v, color, label in [(0, C_CTRL, "Control"), (1, C_ARDS, "ARDS")]:
        mask = y_train == v
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=color, alpha=0.6, s=30, label=label)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Stage 1 — PCA of RF-selected proteins", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("stage1_pca.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Top pathway boxplots
    if len(sig_validation) > 0:
        pathway_genes_str = sig_validation.nsmallest(
            1, "Adjusted P-value").iloc[0]["Genes"]
        top_proteins = [g.strip() for g in pathway_genes_str.split(";")
                        if g.strip() in pat_renamed.columns][:12]

        if top_proteins:
            fig, axes = plt.subplots(3, 4, figsize=(16, 10))
            axes = axes.flatten()
            for i, gene in enumerate(top_proteins):
                ax        = axes[i]
                d_ctrl    = zscored_all.loc[y_train==0, gene]
                d_ards    = zscored_all.loc[y_train==1, gene]
                ax.boxplot([d_ctrl, d_ards], labels=["Control","ARDS"],
                           patch_artist=True,
                           boxprops=dict(facecolor="lightgrey"),
                           medianprops=dict(color="black", linewidth=2))
                for j, (data, color) in enumerate([(d_ctrl, C_CTRL),
                                                    (d_ards, C_ARDS)]):
                    x = np.random.normal(j+1, 0.06, size=len(data))
                    ax.scatter(x, data, alpha=0.5, s=18, color=color, zorder=3)
                ax.set_title(gene, fontweight="bold", fontsize=9)
                ax.set_ylabel("Z-score", fontsize=7)
                ax.axhline(0, color="grey", linestyle="--", alpha=0.4)
            for ax in axes[len(top_proteins):]:
                ax.set_visible(False)
            plt.suptitle("Stage 1 — Top pathway proteins: ARDS vs Control",
                         fontweight="bold")
            plt.tight_layout()
            plt.savefig("stage1_boxplots.png", dpi=150, bbox_inches="tight")
            plt.show()

    return zscored_all


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — MECHANISM
# ══════════════════════════════════════════════════════════════════════════════

def stage2_mechanism(pat_renamed, y_train, background,
                     gene_sets=GENE_SETS, z_threshold=Z_THRESHOLD):
    print("\n" + "="*65)
    print("STAGE 2 — Mechanism: per-ARDS-patient enrichment vs sepsis baseline")
    print("="*65)

    ards_idx    = y_train[y_train==1].index
    control_idx = y_train[y_train==0].index

    ards_pat    = pat_renamed.loc[ards_idx]
    control_pat = pat_renamed.loc[control_idx]

    print(f"ARDS patients    : {len(ards_pat)}")
    print(f"Control patients : {len(control_pat)}")

    # Z-score relative to control
    ctrl_mean = control_pat.mean(axis=0)
    ctrl_std  = control_pat.std(axis=0).replace(0, 1)
    ards_z    = (ards_pat - ctrl_mean) / ctrl_std

    # Tune threshold info
    print("\nThreshold scan:")
    for z in [1.0, 0.5, 0.0, -0.5]:
        hits = [(ards_z.loc[p] > z).sum() for p in ards_z.index]
        print(f"  Z={z:5.2f} — mean: {np.mean(hits):.1f}  "
              f"min: {np.min(hits)}  max: {np.max(hits)}  "
              f"patients>=5: {sum(h>=5 for h in hits)}")

    # Build hit lists
    hit_lists = {}
    for patient in ards_z.index:
        row       = ards_z.loc[patient]
        hit_genes = row[row > z_threshold].index.tolist()
        hit_lists[patient] = hit_genes

    # Run Enrichr
    enrichr_results = {}
    for i, (pid, prot_list) in enumerate(hit_lists.items()):
        if len(prot_list) < 5:
            continue
        if i % 10 == 0:
            print(f"  Enrichr: {i}/{len(hit_lists)} ARDS patients...")
        enr = gp.enrichr(
            gene_list=prot_list, gene_sets=gene_sets,
            background=background, outdir=None, cutoff=1.0,
        )
        enrichr_results[pid] = enr.results.copy()

    print(f"Enrichr complete: {len(enrichr_results)} ARDS patients")

    # Build score matrix
    records = []
    for pid, res in enrichr_results.items():
        row = {"Patient": pid}
        for _, r in res.iterrows():
            row[r["Term"]] = pathway_score_fn(r["Adjusted P-value"])
        records.append(row)

    score_matrix = pd.DataFrame(records).set_index("Patient").fillna(0)
    print(f"Score matrix: {score_matrix.shape}")
    print(f"Pathways with signal: {(score_matrix > 0).any().sum()}")

    # Filter
    min_pts  = max(2, int(MIN_PATIENTS_PCT * len(score_matrix)))
    mask     = (score_matrix > 0).sum(axis=0) >= min_pts
    score_matrix = score_matrix.loc[:, mask]
    print(f"Pathways after filtering (>={MIN_PATIENTS_PCT*100:.0f}%): "
          f"{score_matrix.shape[1]}")

    return ards_z, enrichr_results, score_matrix


def stage2_plots(ards_z, enrichr_results, y_train, severities):
    """Stage 2 visualisations."""
    sev_colors = SEV_PALETTE
    sev_order  = ["Mild", "Moderate", "Severe"]

    # Significant pathway count per patient
    n_sig = {pid: (res["Adjusted P-value"] < FDR_CUTOFF).sum()
             for pid, res in enrichr_results.items()}
    n_sig_df = pd.DataFrame({
        "Patient" : list(n_sig.keys()),
        "N_sig"   : list(n_sig.values()),
        "Severity": severities.reindex(list(n_sig.keys())).values
    }).sort_values("N_sig", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bar_colors = [sev_colors.get(s, "#cccccc") for s in n_sig_df["Severity"]]
    axes[0].bar(range(len(n_sig_df)), n_sig_df["N_sig"],
                color=bar_colors, alpha=0.85, edgecolor="white")
    axes[0].set_xlabel("Patient (sorted)")
    axes[0].set_ylabel("Significant pathways")
    axes[0].set_title("Stage 2 — Significant pathways per ARDS patient",
                      fontweight="bold")
    axes[0].legend(handles=[Patch(color=sev_colors[s], label=s)
                             for s in sev_order if s in sev_colors])

    # Pathway frequency + score bubble plot
    pathway_freq, pathway_score = {}, {}
    for pid, res in enrichr_results.items():
        sig = res[res["Adjusted P-value"] < FDR_CUTOFF]
        for _, row in sig.iterrows():
            pathway_freq[row["Term"]]  = pathway_freq.get(row["Term"], 0) + 1
            sc = -np.log10(max(row["Adjusted P-value"], 1e-300))
            pathway_score.setdefault(row["Term"], []).append(sc)

    if pathway_freq:
        pw_summary = pd.DataFrame({
            "Term"      : list(pathway_freq.keys()),
            "N_patients": list(pathway_freq.values()),
            "Mean_score": [np.mean(v) for v in pathway_score.values()],
        }).sort_values("N_patients", ascending=False).head(20)

        sc = axes[1].scatter(
            pw_summary["N_patients"], range(len(pw_summary)),
            s=pw_summary["Mean_score"] * 40,
            c=pw_summary["Mean_score"], cmap="YlOrRd",
            alpha=0.85, edgecolors="grey", linewidth=0.4
        )
        axes[1].set_yticks(range(len(pw_summary)))
        axes[1].set_yticklabels(pw_summary["Term"].str[:50], fontsize=7)
        axes[1].set_xlabel("N ARDS patients with significant enrichment")
        axes[1].set_title("Stage 2 — Most common ARDS pathways",
                          fontweight="bold")
        plt.colorbar(sc, ax=axes[1], label="Mean -log10(adj p)")

    plt.tight_layout()
    plt.savefig("stage2_overview.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Z-score heatmap
    sev_order_map = {"Mild": 0, "Moderate": 1, "Severe": 2}
    sort_idx = (severities.reindex(ards_z.index)
                .map(sev_order_map).fillna(-1)
                .sort_values().index)
    protein_order = ards_z.mean(axis=0).sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(18, 7))
    sns.heatmap(ards_z.loc[sort_idx, protein_order],
                cmap="RdBu_r", center=0, vmin=-3, vmax=3,
                ax=ax, xticklabels=True, yticklabels=True,
                cbar_kws={"label": "Z-score vs control"},
                linewidths=0)
    ax.set_title("Stage 2 — ARDS patients × proteins (z-score vs control)",
                 fontweight="bold")
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    ax.tick_params(axis="y", labelsize=6)
    plt.tight_layout()
    plt.savefig("stage2_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — ENDOTYPING
# ══════════════════════════════════════════════════════════════════════════════

def stage3_endotyping(score_matrix, severities, max_k=5):
    print("\n" + "="*65)
    print("STAGE 3 — Endotyping: clustering ARDS patients")
    print("="*65)

    cluster_matrix = score_matrix.copy()
    score_scaled   = StandardScaler().fit_transform(cluster_matrix)
    dist_sq        = squareform(pdist(score_scaled, metric="euclidean"))

    # Silhouette
    print("Silhouette scores:")
    sil_scores = {}
    for k in range(2, max_k+1):
        clust = AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage="average"
        ).fit(dist_sq)
        sil = silhouette_score(dist_sq, clust.labels_, metric="precomputed")
        sil_scores[k] = sil
        print(f"  k={k}  silhouette={sil:.4f}")

    best_k = max(sil_scores, key=sil_scores.get)
    print(f"\nBest k: {best_k}")

    # Silhouette plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(sil_scores.keys()), list(sil_scores.values()),
            "o-", color="#E74C3C", linewidth=2)
    ax.set_xlabel("Number of endotypes (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Optimal number of endotypes", fontweight="bold")
    ax.axvline(best_k, color="grey", linestyle="--", alpha=0.5)
    ax.set_xticks(list(sil_scores.keys()))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("stage3_silhouette.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Final clustering
    final = AgglomerativeClustering(
        n_clusters=best_k, metric="precomputed", linkage="average"
    ).fit(dist_sq)

    score_matrix = score_matrix.copy()
    score_matrix["Endotype"]     = final.labels_
    score_matrix["ards_severity"]= severities.reindex(score_matrix.index)

    # Crosstable
    print("\nSeverity × Endotype:")
    print(pd.crosstab(score_matrix["ards_severity"],
                      score_matrix["Endotype"], margins=True))

    # Profiles
    profiles = (score_matrix.drop(columns="ards_severity")
                .groupby("Endotype").mean().T)
    print("\nTop 8 pathways per endotype:")
    for e in sorted(score_matrix["Endotype"].unique()):
        n   = (score_matrix["Endotype"] == e).sum()
        top = profiles[e].sort_values(ascending=False).head(8)
        print(f"\n  Endotype {e}  (n={n}):")
        for term, s in top.items():
            if s > 0:
                print(f"    {s:.2f}  {term[:60]}")

    return score_matrix, profiles, best_k


def stage3_plots(score_matrix, profiles, ards_z, severities):
    """Stage 3 visualisations."""
    endotypes = score_matrix["Endotype"]
    plot_sc   = score_matrix.drop(
        columns=[c for c in ["Endotype","ards_severity"] if c in score_matrix.columns])

    # ── Heatmap: mean pathway score per endotype ───────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, len(profiles.columns)*2), 6))
    sns.heatmap(profiles, cmap="YlOrRd", ax=ax,
                annot=True, fmt=".1f", annot_kws={"size": 7},
                linewidths=0.3,
                cbar_kws={"label": "Mean -log10(adj p)"})
    ax.set_title("Stage 3 — Mean pathway score per endotype",
                 fontweight="bold")
    ax.tick_params(axis="y", labelsize=6)
    plt.tight_layout()
    plt.savefig("stage3_pathway_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── PCA coloured by endotype and severity ──────────────────────────────────
    X_scaled  = StandardScaler().fit_transform(plot_sc)
    pca       = PCA(n_components=2)
    embedding = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for e, color in ENDOTYPE_PALETTE.items():
        mask = endotypes == e
        if mask.sum() == 0: continue
        axes[0].scatter(embedding[mask, 0], embedding[mask, 1],
                        c=color, s=80, alpha=0.85,
                        label=f"Endotype {e} (n={mask.sum()})",
                        edgecolors="white", linewidth=0.5)
    axes[0].set_title("PCA — coloured by endotype", fontweight="bold")
    axes[0].legend(fontsize=8)

    sev_order = ["Mild", "Moderate", "Severe"]
    for sev in sev_order:
        mask = score_matrix["ards_severity"] == sev
        if mask.sum() == 0: continue
        axes[1].scatter(embedding[mask, 0], embedding[mask, 1],
                        c=SEV_PALETTE[sev], s=80, alpha=0.85,
                        label=f"{sev} (n={mask.sum()})",
                        edgecolors="white", linewidth=0.5)
    axes[1].set_title("PCA — coloured by severity", fontweight="bold")
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

    plt.suptitle("Stage 3 — ARDS patients in pathway space",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig("stage3_pca.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── Severity stacked bar ───────────────────────────────────────────────────
    cross = pd.crosstab(score_matrix["Endotype"],
                        score_matrix["ards_severity"])
    cross = cross.reindex(columns=[s for s in sev_order if s in cross.columns])

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom  = np.zeros(len(cross))
    for sev in cross.columns:
        vals = cross[sev].values
        bars = ax.bar(
            [f"E{e}\n(n={( endotypes==e).sum()})" for e in cross.index],
            vals, bottom=bottom,
            color=SEV_PALETTE[sev], label=sev,
            alpha=0.85, edgecolor="white"
        )
        for bar, val, bot in zip(bars, vals, bottom):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bot + val/2, str(int(val)),
                        ha="center", va="center",
                        fontsize=9, fontweight="bold")
        bottom += vals
    ax.set_ylabel("Number of patients")
    ax.set_title("Stage 3 — Severity per endotype", fontweight="bold")
    ax.legend(title="Severity")
    plt.tight_layout()
    plt.savefig("stage3_severity.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── Protein z-scores per endotype ─────────────────────────────────────────
    top10_prot = (ards_z.var(axis=0)
                  .sort_values(ascending=False)
                  .head(10).index.tolist())
    z_e = ards_z.copy()
    z_e["Endotype"] = endotypes.reindex(ards_z.index)

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()
    for i, gene in enumerate(top10_prot):
        ax = axes[i]
        for e, color in ENDOTYPE_PALETTE.items():
            mask = z_e["Endotype"] == e
            if mask.sum() == 0: continue
            data = z_e.loc[mask, gene]
            x    = np.random.normal(e, 0.08, size=len(data))
            ax.scatter(x, data, c=color, alpha=0.75, s=40,
                       edgecolors="white", linewidth=0.3)
            ax.plot([e-0.25, e+0.25], [data.mean(), data.mean()],
                    color=color, linewidth=2.5)
        ax.set_xticks(list(ENDOTYPE_PALETTE.keys()))
        ax.set_xticklabels([f"E{e}" for e in ENDOTYPE_PALETTE.keys()],
                           fontsize=7)
        ax.set_title(gene, fontweight="bold", fontsize=9)
        ax.set_ylabel("Z-score", fontsize=7)
        ax.axhline(0, color="grey", linestyle="--", alpha=0.4)
    plt.suptitle("Stage 3 — Top variable proteins per endotype",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig("stage3_proteins.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── Jaccard co-enrichment ─────────────────────────────────────────────────
    cooccur = (plot_sc > 0).astype(int)
    n       = len(plot_sc.columns)
    jaccard = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a = cooccur.iloc[:, i].values
            b = cooccur.iloc[:, j].values
            inter = (a & b).sum()
            union = (a | b).sum()
            jaccard[i, j] = inter / union if union > 0 else 0

    jac_df = pd.DataFrame(jaccard,
                          index=plot_sc.columns,
                          columns=plot_sc.columns)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(jac_df, cmap="Blues", ax=ax,
                square=True, linewidths=0.2,
                cbar_kws={"label": "Jaccard similarity"},
                xticklabels=True, yticklabels=True)
    ax.tick_params(labelsize=6)
    ax.set_title("Stage 3 — Pathway co-enrichment (Jaccard)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig("stage3_jaccard.png", dpi=150, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION — Ensemble
# ══════════════════════════════════════════════════════════════════════════════

def run_ensemble_classifier(score_matrix, pat_renamed, X_test_renamed,
                             y_train, y_test, enrichr_results):
    print("\n" + "="*65)
    print("CLASSIFICATION — Ensemble classifier")
    print("="*65)

    endotype_means = (
        score_matrix.drop(columns=["ards_severity"], errors="ignore")
        .groupby("Endotype").mean()
    )

    # Get genes per endotype
    def get_endotype_genes(e, top_n=3):
        top_pathways = (endotype_means.loc[e]
                        .sort_values(ascending=False)
                        .head(top_n).index.tolist())
        top_pathways = [p for p in top_pathways
                        if endotype_means.loc[e, p] > 0]
        all_genes = []
        for pathway in top_pathways:
            best_score, best_genes = 0, []
            for pid, res in enrichr_results.items():
                row = res[res["Term"] == pathway]
                if len(row) == 0: continue
                sc = -np.log10(max(row["Adjusted P-value"].values[0], 1e-300))
                if sc > best_score:
                    best_score = sc
                    best_genes = [g.strip() for g in
                                  row["Genes"].values[0].split(";")]
            avail = [g for g in best_genes
                     if g in pat_renamed.columns and g in X_test_renamed.columns]
            all_genes.extend(avail)
        return list(dict.fromkeys(all_genes))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def base_clf():
        return Pipeline([
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", C=0.1,
                max_iter=2000, solver="saga", random_state=42))
        ])

    train_probs, test_probs = {}, {}

    for e in sorted(score_matrix["Endotype"].unique()):
        gene_list = get_endotype_genes(e)
        if len(gene_list) < 2:
            gene_list = [g for g in pat_renamed.columns
                         if g in X_test_renamed.columns]
        print(f"  Endotype {e}: {len(gene_list)} genes", end=" ")

        X_tr = pat_renamed[gene_list]
        X_te = X_test_renamed[gene_list]

        oof = cross_val_predict(
            base_clf(), X_tr, y_train, cv=cv,
            method="predict_proba", n_jobs=-1
        )[:, 1]
        train_probs[f"E{e}"] = oof

        model = base_clf()
        model.fit(X_tr, y_train)
        test_probs[f"E{e}"] = model.predict_proba(X_te)[:, 1]

        print(f"OOF AUC={roc_auc_score(y_train, oof):.3f}  "
              f"Test AUC={roc_auc_score(y_test, test_probs[f'E{e}']):.3f}")

    X_meta_train = pd.DataFrame(train_probs, index=pat_renamed.index)
    X_meta_test  = pd.DataFrame(test_probs,  index=X_test_renamed.index)

    # Meta-classifiers
    meta_configs = {
        "LR meta"  : LogisticRegression(class_weight="balanced", C=1.0,
                                         max_iter=1000, random_state=42),
        "RF meta"  : RandomForestClassifier(n_estimators=500, max_depth=3,
                                             class_weight="balanced",
                                             random_state=42),
        "Max prob" : None,
        "Mean prob": None,
    }

    meta_results = []
    print("\nMeta-classifier results:")
    for name, clf in meta_configs.items():
        if name == "Max prob":
            tr_pred = X_meta_train.max(axis=1).values
            te_pred = X_meta_test.max(axis=1).values
        elif name == "Mean prob":
            tr_pred = X_meta_train.mean(axis=1).values
            te_pred = X_meta_test.mean(axis=1).values
        else:
            tr_pred = cross_val_predict(
                clf, X_meta_train, y_train, cv=cv,
                method="predict_proba", n_jobs=-1
            )[:, 1]
            clf.fit(X_meta_train, y_train)
            te_pred = clf.predict_proba(X_meta_test)[:, 1]

        te_label = (te_pred >= 0.5).astype(int)
        meta_results.append({
            "Meta": name,
            "OOF AUC" : round(roc_auc_score(y_train, tr_pred), 3),
            "Test AUC": round(roc_auc_score(y_test,  te_pred),  3),
            "Test F1" : round(f1_score(y_test, te_label),       3),
            "BalAcc"  : round(balanced_accuracy_score(y_test, te_label), 3),
            "_pred"   : te_pred,
        })
        print(f"  {name:12s}  OOF={meta_results[-1]['OOF AUC']}  "
              f"Test={meta_results[-1]['Test AUC']}  "
              f"F1={meta_results[-1]['Test F1']}")

    meta_df = pd.DataFrame(meta_results)

    # ── ROC plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for (e_label, probs), color in zip(test_probs.items(),
                                        list(ENDOTYPE_PALETTE.values())):
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"Endotype {e_label[-1]} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1],"k--", alpha=0.3)
    ax.set_title("Base classifiers (per endotype)", fontweight="bold")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=8)

    ax = axes[1]
    meta_colors = ["#1a6eb5", "#9b1b30", "#555", "#222"]
    for (_, row), color in zip(meta_df.iterrows(), meta_colors):
        fpr, tpr, _ = roc_curve(y_test, row["_pred"])
        ax.plot(fpr, tpr, linewidth=2, color=color,
                label=f"{row['Meta']} (AUC={row['Test AUC']})")
    ax.plot([0,1],[0,1],"k--", alpha=0.3)
    ax.set_title("Meta-classifiers (ensemble)", fontweight="bold")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=8)

    plt.suptitle("Ensemble classifier — unseen.csv", fontweight="bold")
    plt.tight_layout()
    plt.savefig("ensemble_roc.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── Endotype probability heatmap ───────────────────────────────────────────
    test_prob_df = X_meta_test.copy()
    test_prob_df["ARDS_true"] = y_test.values
    test_prob_df = test_prob_df.sort_values("ARDS_true")

    fig, ax = plt.subplots(figsize=(7, 10))
    im = ax.imshow(test_prob_df.drop(columns="ARDS_true").values,
                   aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(X_meta_test.columns)))
    ax.set_xticklabels([f"Endotype {c[-1]}" for c in X_meta_test.columns],
                       fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(test_prob_df)))
    ax.set_yticklabels(
        [f"{'► ARDS' if v else '  Ctrl'}" for v in test_prob_df["ARDS_true"]],
        fontsize=6,
        color=["#c0392b" if v else "#2980b9"
               for v in test_prob_df["ARDS_true"]])
    plt.colorbar(im, ax=ax, label="P(ARDS | endotype genes)")
    ax.set_title("Test patients — endotype probability vectors",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig("ensemble_probs.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\nBest: {meta_df.loc[meta_df['Test AUC'].idxmax(), 'Meta']} "
          f"(AUC={meta_df['Test AUC'].max()})")

    return meta_df.drop(columns="_pred")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(selected_proteins_df, all_data_df, seen_df, unseen_df, anno,
                 importance_col="importance",
                 severity_df=None, severity_col="ards_severity",
                 severity_id_col="SampleId"):
    """
    Run the full three-stage ARDS enrichment pipeline.

    Parameters
    ----------
    selected_proteins_df : DataFrame with columns ["proteins", importance_col]
    all_data_df          : All patients — rows=patients, seq cols + label col
    seen_df              : Training split
    unseen_df            : Test split
    anno                 : SomaScan annotation DataFrame
    importance_col       : Column name for importance/frequency in selected_proteins_df
    severity_df          : Optional DataFrame with severity info
    severity_col         : Column name for severity
    severity_id_col      : Patient ID column in severity_df
    """

    print("Building sequence → gene mapping...")
    seq_to_gene = build_seq_to_gene(anno)

    # ── Prepare patient tables ─────────────────────────────────────────────────
    print("Preparing patient tables...")
    pat_all,    labels_all    = prepare_patient_df(all_data_df,  LABEL_COL, SAMPLE_ID_COL)
    pat_seen,   labels_seen   = prepare_patient_df(seen_df,      LABEL_COL, SAMPLE_ID_COL)
    pat_unseen, labels_unseen = prepare_patient_df(unseen_df,    LABEL_COL, SAMPLE_ID_COL)

    # Rename seq → gene
    print("Renaming proteins to gene symbols...")
    pat_seen_renamed   = rename_df_columns(pat_seen,   seq_to_gene)
    pat_unseen_renamed = rename_df_columns(pat_unseen, seq_to_gene)

    shared_cols = [c for c in pat_seen_renamed.columns
                   if c in pat_unseen_renamed.columns]
    pat_seen_renamed   = pat_seen_renamed[shared_cols]
    pat_unseen_renamed = pat_unseen_renamed[shared_cols]
    print(f"Shared gene columns: {len(shared_cols)}")

    y_train = labels_seen.astype(int)
    y_test  = labels_unseen.astype(int)

    # ── Robust proteins from RF ────────────────────────────────────────────────
    print(f"\nSelecting robust proteins (importance >= {IMPORTANCE_CUTOFF})...")
    robust_seqs = set(
        selected_proteins_df[
            selected_proteins_df[importance_col] >= IMPORTANCE_CUTOFF
        ]["proteins"].tolist()
    )
    robust_genes = list(set(
        seq_to_gene[s] for s in robust_seqs
        if s in seq_to_gene and seq_to_gene[s] in shared_cols
    ))
    background = list(set(
        seq_to_gene[s] for s in pat_seen.columns
        if s in seq_to_gene and seq_to_gene[s] in shared_cols
    ))
    print(f"Robust proteins  : {len(robust_genes)}")
    print(f"Background genes : {len(background)}")

    # Restrict seen to robust proteins
    robust_seen = pat_seen_renamed[
        [g for g in robust_genes if g in pat_seen_renamed.columns]
    ]

    # ── Severity ───────────────────────────────────────────────────────────────
    if severity_df is not None:
        severities = (severity_df.set_index(severity_id_col)[severity_col]
                      if severity_id_col in severity_df.columns
                      else severity_df[severity_col])
    else:
        # Try to find severity in all_data_df
        if severity_col in all_data_df.columns:
            tmp = all_data_df.copy()
            if SAMPLE_ID_COL in tmp.columns:
                tmp = tmp.set_index(SAMPLE_ID_COL)
            severities = tmp[severity_col]
        else:
            print(f"Warning: '{severity_col}' not found — severity plots skipped")
            severities = pd.Series(np.nan, index=pat_seen_renamed.index)

    # ── Stage 1 ────────────────────────────────────────────────────────────────
    sig_validation, _ = stage1_validation(robust_genes, background)
    zscored_all       = stage1_plots(robust_seen, y_train, sig_validation)

    # ── Stage 2 ────────────────────────────────────────────────────────────────
    sev_ards = severities.reindex(
        y_train[y_train==1].index
    )
    ards_z, enrichr_results, score_matrix = stage2_mechanism(
        robust_seen, y_train, background
    )
    stage2_plots(ards_z, enrichr_results, y_train, sev_ards)

    # ── Stage 3 ────────────────────────────────────────────────────────────────
    score_matrix, profiles, best_k = stage3_endotyping(
        score_matrix, sev_ards
    )
    stage3_plots(score_matrix, profiles, ards_z, sev_ards)

    # ── Ensemble classifier ────────────────────────────────────────────────────
    meta_results = run_ensemble_classifier(
        score_matrix, robust_seen, pat_unseen_renamed,
        y_train, y_test, enrichr_results
    )

    print("\n" + "="*65)
    print("Pipeline complete.")
    print("="*65)

    return {
        "sig_validation"  : sig_validation,
        "ards_zscored"    : ards_z,
        "enrichr_results" : enrichr_results,
        "score_matrix"    : score_matrix,
        "endotype_profiles": profiles,
        "best_k"          : best_k,
        "meta_results"    : meta_results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — called from notebook
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ── Set your inputs here ──────────────────────────────────────────────────
    # These variables must exist in your notebook before running %run pipeline.py
    # or you can set them directly here:
    #
    # selected_proteins_df = pd.read_csv("rf_selected_proteins.csv")
    # all_data_df          = pd.read_csv("../data/processed/all.csv")
    # seen_df              = pd.read_csv("../data/processed/seen.csv")
    # unseen_df            = pd.read_csv("../data/processed/unseen.csv")
    # anno                 = pd.read_csv("../data/processed/somalogic_annotation.csv")

    results = run_pipeline(
        selected_proteins_df = selected_proteins_df,
        all_data_df          = all_data_df,
        seen_df              = seen_df,
        unseen_df            = unseen_df,
        anno                 = anno,
        importance_col       = "importance",   # or "frequency" for ttest version
        severity_df          = None,           # or pass clean_data if needed
        severity_col         = "ards_severity",
        severity_id_col      = "SampleId",
    )