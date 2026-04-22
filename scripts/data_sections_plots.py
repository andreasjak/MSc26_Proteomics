"""
Data Section — Plots for Proteomics ARDS Thesis
================================================
Produces publication-quality figures for the Data section.
All figures are saved as high-resolution PDFs (vector) suitable
for direct inclusion in LaTeX via \includegraphics.

Usage
-----
    python data_section_plots.py

Output files (in ./figures/)
-----------------------------
    fig_cohort_flowchart.pdf      – CONSORT-style cohort flow
    fig_cohort_overview.pdf       – Cohort bar + sex grouped bar (2-panel)
    fig_age_distributions.pdf     – 2×2 age histogram grid
    fig_icu_site.pdf              – ICU-site stacked bar + ARDS-rate overlay
    fig_protein_distributions.pdf – 3-panel protein expression distributions
    fig_pca.pdf                   – PCA biplot coloured by ARDS / site / sex

Adapt the DATA LOADING block at the top to point at your real dataframe.
All column name assumptions are documented inline.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

os.makedirs("figures", exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "legend.frameon":     False,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

# Colour palette  (colourblind-friendly)
C_ARDS    = "#C0392B"   # red
C_NON     = "#2471A3"   # blue
C_MOD     = "#E67E22"   # orange  (moderate ARDS)
C_SEV     = "#8E44AD"   # purple  (severe ARDS)
C_MEN     = "#1A5276"   # dark blue
C_WOMEN   = "#A93226"   # dark red
C_SITE    = ["#2E86AB", "#A23B72", "#F18F01"]  # per ICU site
ALPHA_LO  = 0.45
ALPHA_HI  = 0.80

# ── DATA LOADING ──────────────────────────────────────────────────────────────
# Replace this block with your real dataframe.
# Expected columns:
#   ards        : bool  (True = ARDS moderate/severe)
#   female      : bool  (True = female)
#   age         : float (years)
#   IVAavd      : str   (ICU unit name, e.g. "Lund")
#   ards_sev    : str   ("Moderate" | "Severe" | NaN for non-ARDS)
#   <protein_*> : float (SomaScan RFU values — all remaining columns)
#
# Example stub with synthetic data matching the reported cohort statistics:
np.random.seed(42)
n_total = 409
n_ards  = 65
n_non   = 344

def _gauss_age(mean, sd, n):
    return np.clip(np.random.normal(mean, sd, n), 20, 96)

ards_idx = np.zeros(n_total, dtype=bool)
ards_idx[:n_ards] = True
np.random.shuffle(ards_idx)

female = np.zeros(n_total, dtype=bool)
female[:171] = True
np.random.shuffle(female)

age = np.where(ards_idx,
               _gauss_age(64, 11, n_total),
               _gauss_age(66, 12, n_total))

site_labels = ["Helsingborg", "Lund", "Malmö"]
site_counts = [134, 116, 158]  # total per site
site_ards   = [10,  26,  29]   # ARDS per site
site_col    = np.empty(n_total, dtype=object)
ptr = 0
for s, cnt in zip(site_labels, site_counts):
    site_col[ptr:ptr+cnt] = s
    ptr += cnt
np.random.shuffle(site_col)

ards_sev = np.where(ards_idx,
                    np.where(np.random.random(n_total) < 44/65, "Moderate", "Severe"),
                    "None")

# Protein block — 200 synthetic proteins (log-normal)
n_proteins = 200
prot_names = [f"PROT_{i:04d}" for i in range(n_proteins)]
mu    = np.random.uniform(7, 9, n_proteins)
sigma = np.random.uniform(0.8, 1.4, n_proteins)
prot_matrix = np.exp(
    np.random.normal(mu, sigma, (n_total, n_proteins))
    + np.outer(ards_idx.astype(float), np.random.normal(0.15, 0.4, n_proteins))
)

df = pd.DataFrame(prot_matrix, columns=prot_names)
df["ards"]     = ards_idx
df["female"]   = female
df["age"]      = age
df["IVAavd"]   = site_col
df["ards_sev"] = ards_sev

# Convenience sub-frames
ards_df     = df[df["ards"] == True]
non_ards_df = df[df["ards"] == False]

features = prot_names  # replace with your get_protein_features(df) output


# ═════════════════════════════════════════════════════════════════════════════
# FIG 1 — CONSORT-style cohort flow diagram
# ═════════════════════════════════════════════════════════════════════════════
def fig_cohort_flowchart():
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    BOX_W, BOX_H = 4.2, 0.75
    CX = 5.0   # centre x of main boxes
    EX = 8.8   # centre x of exclusion boxes

    def box(cx, cy, text, color="#2C3E50", facecolor="#EBF5FB", width=BOX_W):
        rect = mpatches.FancyBboxPatch(
            (cx - width/2, cy - BOX_H/2), width, BOX_H,
            boxstyle="round,pad=0.08", linewidth=0.9,
            edgecolor=color, facecolor=facecolor, zorder=3
        )
        ax.add_patch(rect)
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=8.5, zorder=4, color="#1A252F")

    def arrow(x, y_start, y_end):
        ax.annotate("", xy=(x, y_end + BOX_H/2 + 0.05),
                    xytext=(x, y_start - BOX_H/2 - 0.05),
                    arrowprops=dict(arrowstyle="-|>", color="#555",
                                   lw=0.9, mutation_scale=10),
                    zorder=2)

    def excl_box(cy, text):
        rect = mpatches.FancyBboxPatch(
            (EX - 2.2, cy - BOX_H/2), 4.4 - 0.2, BOX_H,
            boxstyle="round,pad=0.08", linewidth=0.8,
            edgecolor="#C0392B", facecolor="#FDEDEC", zorder=3
        )
        ax.add_patch(rect)
        ax.text(EX - 0.1, cy, text, ha="center", va="center",
                fontsize=8, zorder=4, color="#922B21")
        # horizontal dashed arrow from main spine
        ax.annotate("", xy=(EX - 2.4, cy),
                    xytext=(CX + BOX_W/2, cy),
                    arrowprops=dict(arrowstyle="-|>", color="#C0392B",
                                   lw=0.8, linestyle="dashed",
                                   mutation_scale=9),
                    zorder=2)

    # Main boxes (top → bottom)
    ys = [9.2, 7.4, 5.6, 3.8, 2.0]
    texts = [
        "Initial cohort\nn = 1 449",
        "Sepsis patients\nn = 418",
        "Mild ARDS excluded\nn = 409",
        "QC-passed samples\nn = 409  (final cohort)",
        "ARDS (mod/severe) = 65     Non-ARDS = 343",
    ]
    face_colors = ["#EBF5FB", "#EBF5FB", "#EBF5FB", "#D5F5E3", "#FDFEFE"]
    edge_colors = ["#2C3E50", "#2C3E50", "#2C3E50", "#1E8449", "#555"]

    for i, (y, txt, fc, ec) in enumerate(zip(ys, texts, face_colors, edge_colors)):
        box(CX, y, txt, color=ec, facecolor=fc)

    # Arrows between main boxes
    for i in range(len(ys) - 1):
        arrow(CX, ys[i], ys[i+1])

    # Exclusion boxes
    excl_data = [
        (7.4, "Excluded: no sepsis\nn = 1 031"),
        (5.6, "Excluded: mild ARDS\nn = 9"),
    ]
    for cy, txt in excl_data:
        excl_box(cy, txt)

    ax.set_title("Cohort construction", fontsize=11, fontweight="normal",
                 pad=8, loc="left")
    fig.tight_layout()
    fig.savefig("figures/fig_cohort_flowchart.pdf")
    plt.close(fig)
    print("Saved: fig_cohort_flowchart.pdf")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 2 — Cohort overview: ARDS sizes + ARDS severity + sex breakdown
# ═════════════════════════════════════════════════════════════════════════════
def fig_cohort_overview():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))

    # ── Panel A: cohort sizes ─────────────────────────────────────────────
    ax = axes[0]
    labels  = ["Non-ARDS", "ARDS\n(mod/severe)"]
    counts  = [len(non_ards_df), len(ards_df)]
    colors  = [C_NON, C_ARDS]
    bars = ax.bar(labels, counts, color=colors, alpha=ALPHA_HI,
                  edgecolor="white", linewidth=0.8, width=0.5)
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
                str(n), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Number of patients")
    ax.set_title("(a) Cohort sizes", loc="left", fontsize=10)
    ax.set_ylim(0, max(counts) * 1.18)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(50))

    # ── Panel B: ARDS severity ────────────────────────────────────────────
    ax = axes[1]
    sev_counts = ards_df["ards_sev"].value_counts().reindex(["Moderate","Severe"])
    bars = ax.bar(sev_counts.index, sev_counts.values,
                  color=[C_MOD, C_SEV], alpha=ALPHA_HI,
                  edgecolor="white", linewidth=0.8, width=0.5)
    for bar, n in zip(bars, sev_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(n), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Number of ARDS patients")
    ax.set_title("(b) ARDS severity", loc="left", fontsize=10)
    ax.set_ylim(0, sev_counts.max() * 1.22)

    # ── Panel C: sex by ARDS status ───────────────────────────────────────
    ax = axes[2]
    grp_labels = ["Non-ARDS\nWomen", "Non-ARDS\nMen",
                  "ARDS\nWomen",     "ARDS\nMen"]
    grp_vals = [
        non_ards_df["female"].sum(),
        (~non_ards_df["female"]).sum(),
        ards_df["female"].sum(),
        (~ards_df["female"]).sum(),
    ]
    grp_colors = [C_NON, C_NON, C_ARDS, C_ARDS]
    grp_alpha  = [ALPHA_LO, ALPHA_HI, ALPHA_LO, ALPHA_HI]
    xs = np.arange(4)
    for x, v, c, a in zip(xs, grp_vals, grp_colors, grp_alpha):
        ax.bar(x, v, color=c, alpha=a, edgecolor="white", linewidth=0.8, width=0.6)
        ax.text(x, v + 0.5, str(v), ha="center", va="bottom", fontsize=9,
                fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(grp_labels, fontsize=8.5)
    ax.set_ylabel("Number of patients")
    ax.set_title("(c) Sex by ARDS status", loc="left", fontsize=10)
    ax.set_ylim(0, max(grp_vals) * 1.18)

    # shared legend for panel C colour convention
    leg = [mpatches.Patch(color=C_NON, alpha=0.6, label="Non-ARDS"),
           mpatches.Patch(color=C_ARDS, alpha=0.7, label="ARDS"),
           mpatches.Patch(color="#aaa",  alpha=ALPHA_LO, label="Women (lighter)"),
           mpatches.Patch(color="#aaa",  alpha=ALPHA_HI, label="Men (darker)")]
    axes[2].legend(handles=leg, fontsize=7.5, loc="upper right",
                   handlelength=1.2, handletextpad=0.5)

    fig.suptitle("Cohort overview", fontsize=11, x=0.02, ha="left", y=1.01)
    fig.tight_layout()
    fig.savefig("figures/fig_cohort_overview.pdf")
    plt.close(fig)
    print("Saved: fig_cohort_overview.pdf")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 3 — Age distributions (2 × 2 grid)
# ═════════════════════════════════════════════════════════════════════════════
def fig_age_distributions():
    bins = np.arange(20, 97, 3)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)

    def _hist(ax, data_list, labels, colors, alphas, title, mean_colors=None):
        for vals, lbl, col, alp in zip(data_list, labels, colors, alphas):
            ax.hist(vals, bins=bins, color=col, alpha=alp,
                    edgecolor="white", linewidth=0.4, label=lbl)
        if mean_colors is None:
            mean_colors = colors
        for vals, lbl, mc in zip(data_list, labels, mean_colors):
            m = np.nanmean(vals)
            ax.axvline(m, color=mc, linewidth=1.4, linestyle="--",
                       label=f"Mean {lbl.split()[0]} = {m:.0f} yr")
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Number of patients")
        ax.set_title(title, loc="left", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_xlim(18, 96)

    _hist(axes[0, 0],
          [df["age"]],
          ["All patients"],
          [C_NON], [0.70],
          "(a) All patients")

    _hist(axes[0, 1],
          [non_ards_df["age"], ards_df["age"]],
          ["Non-ARDS", "ARDS"],
          [C_NON, C_ARDS], [0.70, 0.65],
          "(b) ARDS vs Non-ARDS",
          mean_colors=[C_NON, C_ARDS])

    _hist(axes[1, 0],
          [df[~df["female"]]["age"], df[df["female"]]["age"]],
          ["Men", "Women"],
          [C_MEN, C_WOMEN], [0.65, 0.55],
          "(c) Men vs Women",
          mean_colors=[C_MEN, C_WOMEN])

    # 4-group: Non-ARDS men/women + ARDS men/women
    subg = {
        "Non-ARDS men":   non_ards_df[~non_ards_df["female"]]["age"],
        "Non-ARDS women": non_ards_df[non_ards_df["female"]]["age"],
        "ARDS men":       ards_df[~ards_df["female"]]["age"],
        "ARDS women":     ards_df[ards_df["female"]]["age"],
    }
    sc = [C_NON, C_NON, C_ARDS, C_ARDS]
    sa = [0.80, 0.40, 0.80, 0.40]
    _hist(axes[1, 1],
          list(subg.values()), list(subg.keys()),
          sc, sa,
          "(d) ARDS vs Non-ARDS by sex",
          mean_colors=sc)

    fig.suptitle("Age distributions", fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig("figures/fig_age_distributions.pdf")
    plt.close(fig)
    print("Saved: fig_age_distributions.pdf")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 4 — ICU site: stacked bar + ARDS-rate dot overlay
# ═════════════════════════════════════════════════════════════════════════════
def fig_icu_site():
    site_df = (
        df.groupby(["IVAavd", "ards"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={False: "Non-ARDS", True: "ARDS"})
    )
    site_df["Total"]    = site_df.sum(axis=1)
    site_df["ARDS_pct"] = (site_df["ARDS"] / site_df["Total"] * 100)
    site_df = site_df.loc[site_labels]  # canonical order

    fig, ax1 = plt.subplots(figsize=(6.5, 4))

    xs = np.arange(len(site_labels))
    w  = 0.55
    ax1.bar(xs, site_df["Non-ARDS"], width=w,
            color=C_NON, alpha=0.75, edgecolor="white", label="Non-ARDS")
    ax1.bar(xs, site_df["ARDS"], width=w,
            bottom=site_df["Non-ARDS"],
            color=C_ARDS, alpha=0.80, edgecolor="white", label="ARDS")

    # total labels on top of each stack
    for i, (tot, pct) in enumerate(zip(site_df["Total"], site_df["ARDS_pct"])):
        ax1.text(i, tot + 2, f"n={tot}", ha="center", va="bottom",
                 fontsize=8.5, color="#333")

    ax1.set_xticks(xs)
    ax1.set_xticklabels(site_labels)
    ax1.set_ylabel("Number of patients")
    ax1.set_ylim(0, site_df["Total"].max() * 1.18)

    # ARDS rate on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(xs, site_df["ARDS_pct"], color="#E67E22", marker="D",
             markersize=7, linewidth=1.5, label="ARDS rate (%)", zorder=5)
    for i, pct in enumerate(site_df["ARDS_pct"]):
        ax2.text(i + 0.14, pct + 0.4, f"{pct:.1f}%",
                 fontsize=8.5, color="#CA6F1E", va="bottom")
    ax2.set_ylabel("ARDS prevalence (%)", color="#CA6F1E")
    ax2.tick_params(axis="y", colors="#CA6F1E")
    ax2.spines["right"].set_edgecolor("#CA6F1E")
    ax2.set_ylim(0, 40)

    # Combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2,
               loc="upper left", fontsize=8.5)

    ax1.set_title("Patient distribution across ICU sites", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig("figures/fig_icu_site.pdf")
    plt.close(fig)
    print("Saved: fig_icu_site.pdf")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 5 — Protein expression distributions (3 panels)
# ═════════════════════════════════════════════════════════════════════════════
def fig_protein_distributions():
    np.random.seed(0)
    sample_feats = np.random.choice(features, size=min(500, len(features)), replace=False)
    sample_100   = np.random.choice(features, size=min(100, len(features)), replace=False)

    vals_all     = df[sample_feats].values.flatten()
    vals_ards    = ards_df[sample_100].values.flatten()
    vals_non     = non_ards_df[sample_100].values.flatten()
    mean_ards    = ards_df[features].mean(axis=1)
    mean_non     = non_ards_df[features].mean(axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # ── Panel A: overall distribution (log-scale x) ───────────────────────
    ax = axes[0]
    log_vals = np.log10(vals_all[vals_all > 0])
    ax.hist(log_vals, bins=60, color=C_NON, alpha=0.75,
            edgecolor="white", linewidth=0.3, density=True)
    ax.set_xlabel("log₁₀(RFU)")
    ax.set_ylabel("Density")
    ax.set_title("(a) Overall distribution\n(sample of 500 proteins)", loc="left", fontsize=10)

    # ── Panel B: ARDS vs Non-ARDS overlay ─────────────────────────────────
    ax = axes[1]
    log_non  = np.log10(vals_non[vals_non > 0])
    log_ards = np.log10(vals_ards[vals_ards > 0])
    bins_b   = np.linspace(
        min(log_non.min(), log_ards.min()),
        max(log_non.max(), log_ards.max()), 60)
    ax.hist(log_non,  bins=bins_b, color=C_NON,  alpha=0.70,
            edgecolor="white", linewidth=0.3, density=True, label="Non-ARDS")
    ax.hist(log_ards, bins=bins_b, color=C_ARDS, alpha=0.60,
            edgecolor="white", linewidth=0.3, density=True, label="ARDS")
    ax.set_xlabel("log₁₀(RFU)")
    ax.set_ylabel("Density")
    ax.set_title("(b) ARDS vs Non-ARDS\n(sample of 100 proteins)", loc="left", fontsize=10)
    ax.legend()

    # ── Panel C: mean RFU per patient ─────────────────────────────────────
    ax = axes[2]
    log_mn  = np.log10(mean_non[mean_non > 0])
    log_ma  = np.log10(mean_ards[mean_ards > 0])
    bins_c  = np.linspace(min(log_mn.min(), log_ma.min()),
                          max(log_mn.max(), log_ma.max()), 35)
    ax.hist(log_mn, bins=bins_c, color=C_NON,  alpha=0.70,
            edgecolor="white", linewidth=0.4, density=True, label="Non-ARDS")
    ax.hist(log_ma, bins=bins_c, color=C_ARDS, alpha=0.65,
            edgecolor="white", linewidth=0.4, density=True, label="ARDS")
    for vals_p, col in [(log_mn, C_NON), (log_ma, C_ARDS)]:
        ax.axvline(vals_p.mean(), color=col, linewidth=1.3, linestyle="--")
    ax.set_xlabel("log₁₀(mean RFU per patient)")
    ax.set_ylabel("Density")
    ax.set_title("(c) Mean per-patient expression\n(all proteins)", loc="left", fontsize=10)
    ax.legend()

    fig.suptitle("SomaScan protein expression distributions", fontsize=11,
                 x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig("figures/fig_protein_distributions.pdf")
    plt.close(fig)
    print("Saved: fig_protein_distributions.pdf")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 6 — PCA biplot (3 panels: ARDS status / ICU site / Sex)
# ═════════════════════════════════════════════════════════════════════════════
def fig_pca():
    X = df[features].values
    X_log = np.log1p(X)
    X_sc  = StandardScaler().fit_transform(X_log)
    pca   = PCA(n_components=2, random_state=0)
    Z     = pca.fit_transform(X_sc)
    var   = pca.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    def _scatter(ax, groups, title):
        for label, mask, color, marker in groups:
            ax.scatter(Z[mask, 0], Z[mask, 1],
                       c=color, marker=marker, s=18, alpha=0.65,
                       edgecolors="none", label=label, rasterized=True)
        ax.set_xlabel(f"PC 1 ({var[0]:.1f}% var)", fontsize=9)
        ax.set_ylabel(f"PC 2 ({var[1]:.1f}% var)", fontsize=9)
        ax.set_title(title, loc="left", fontsize=10)
        ax.legend(markerscale=1.4, fontsize=8)
        ax.axhline(0, color="#ccc", linewidth=0.5, zorder=0)
        ax.axvline(0, color="#ccc", linewidth=0.5, zorder=0)

    # Panel A — ARDS status
    _scatter(axes[0], [
        ("Non-ARDS", ~df["ards"].values, C_NON,  "o"),
        ("ARDS",      df["ards"].values, C_ARDS, "^"),
    ], "(a) ARDS status")

    # Panel B — ICU site
    site_colors = dict(zip(site_labels, C_SITE))
    site_groups = [(s, df["IVAavd"].values == s, c, "o")
                   for s, c in site_colors.items()]
    _scatter(axes[1], site_groups, "(b) ICU site")

    # Panel C — Sex
    _scatter(axes[2], [
        ("Women", df["female"].values,  C_WOMEN, "o"),
        ("Men",  ~df["female"].values,  C_MEN,   "s"),
    ], "(c) Sex")

    fig.suptitle(
        f"PCA of SomaScan proteome  (PC1={var[0]:.1f}%, PC2={var[1]:.1f}%)",
        fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig("figures/fig_pca.pdf")
    plt.close(fig)
    print("Saved: fig_pca.pdf")


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig_cohort_flowchart()
    fig_cohort_overview()
    fig_age_distributions()
    fig_icu_site()
    fig_protein_distributions()
    fig_pca()
    print("\nAll figures saved to ./figures/")