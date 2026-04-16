# Implementation Guide: Non-Linear Protein Regulation and Higher-Order Interactions in ARDS

This document is the implementation guide for the thesis project. It is written to be consumed by a coding assistant (Claude Code, Copilot) or a human implementer. Each stage is self-contained and has a clear input/output contract. Do the stages in order. Do not skip ahead.

## Project context (read first)

- **Data:** SWECRIT cohort, ~400 patients (growing), SomaScan proteomics with ~11,000 aptamers per patient. Binary outcome: sepsis+ARDS (moderate/severe) vs sepsis-only. ~20% ARDS prevalence.
- **Aim:** identify non-linearly regulated proteins and higher-order protein interactions distinguishing the two groups.
- **Approach:** multiple protein selection methods evaluated under a shared framework (repeated stratified resampling + three validations: simulation, classifier, enrichment).
- **Stability regime:** data-limited. t-test baseline gives top protein at ~72% selection frequency over 50 splits. Expect non-linear methods to be equal or worse. Build the pipeline assuming low stability.

## Repository layout (target)

```
thesis_project/
├── data/
│   ├── raw/                      # SomaScan output, clinical data (read-only)
│   └── processed/                # filtered cohort, cached splits
├── src/
│   ├── config.py                 # paths, K, seed, k values, π, τ, c, etc.
│   ├── data_loading.py           # load + filter cohort, return X, y
│   ├── splits.py                 # generate and cache stratified splits
│   ├── selection/
│   │   ├── base.py               # common interface
│   │   ├── ttest.py
│   │   └── random.py
│   ├── classifiers.py            # fixed LR, RF, XGB configs
│   ├── enrichment.py             # ORA + BES + permutation null
│   ├── simulation.py             # data generator with planted signals
│   ├── metrics.py                # AUC, AUC-PR, recall, FDR helpers
│   └── utils.py                  # logging, IO, seeding
├── scripts/
│   ├── 00_prepare_data.py
│   ├── 01_generate_splits.py
│   ├── 02_run_selection.py
│   ├── 03_run_classifiers.py
│   ├── 04_run_enrichment.py
│   ├── 05_run_simulation.py
│   └── 06_aggregate_results.py
├── notebooks/
├── results/
├── environment.yml
└── README.md
```

## Global conventions

- **Language:** Python 3.11+.
- **Style:** `src/` = importable modules with no side effects on import. `scripts/` = thin CLI entry points that call `src/` and write to `results/`.
- **Data formats:** parquet for tables, pickle only for split index objects, JSON for small config dumps, npz for large numeric arrays. No CSV except for enrichment input/output where required by external tools.
- **Seeding:** every script accepts `--seed` (default from `config.py`). Per-split seeds are derived deterministically from the global seed: `split_seed = global_seed + split_id`.
- **Logging:** every script writes a log file to `results/<stage>/<method>/run.log` with config used, git hash, start/end time, inputs read, outputs written. Use the `logging` module, not `print`.
- **CLI:** every script uses `argparse`. Required convention: `--method <name>` where applicable, `--config <path>` optional override, `--output-dir <path>` optional override.
- **No magic numbers in scripts.** Everything lives in `src/config.py`.
- **Reproducibility:** commit `environment.yml` or `pyproject.toml`. Tag commits when results are generated for the thesis.

## Output format conventions

All results tables use long format (one row per observation). Columns listed per stage below.

## Stage 0: Project scaffolding

**Goal:** create the directory structure

**Steps:**

1. Create the directory tree as listed above. Empty `__init__.py` in every `src/` subdirectory.
2. Update a minimal `README.md` with project description and setup instructions.
3. Update `.gitignore` to exclude `data/raw/`, `data/processed/`, `results/`, `__pycache__/`, `.ipynb_checkpoints/`, `*.pkl`, `*.parquet`.

**Done when:** The directory tree matches the layout above.

## Stage 1: Configuration module

**Goal:** single source of truth for all parameters.

**File:** `src/config.py`

**Contents:**

- Paths: `DATA_RAW`, `DATA_PROCESSED`, `RESULTS_DIR` (absolute or relative to repo root).
- Resampling: `K_SPLITS = 50`, `K_SPLITS_EXPENSIVE = 10`, `TEST_SIZE = 0.2`, `RANDOM_SEED = 42`.
- Top-k values for evaluation: `TOPK_VALUES = [10, 25, 50, 100]`.
- Stability: `STABILITY_PI_PRIMARY = 0.3`, `STABILITY_MIN_SIZE = 20` (fallback to top-k by frequency if stable set below this).
- Enrichment: `BES_CAP_C = 10`, `BES_JACCARD_TAU = 0.5`, `ENRICHMENT_Q_THRESHOLD = 0.05`, `ENRICHMENT_LIBRARIES = ["GO_Biological_Process_2023", "KEGG_2021_Human", "Reactome_2022"]`, `PERMUTATION_B = 1000`.
- Classifiers: dictionaries with the fixed hyperparameters agreed in the methodology (LR, RF, XGB).
- Simulation: `SIM_N = 400`, `SIM_P = 100`, `SIM_CLASS_PREVALENCE = 0.2`, `SIM_SIGNALS_PER_TYPE = 3`, `SIM_EFFECT_SIZES = [0.3, 0.6, 1.0]`, `SIM_REPEATS = 50`.

**Done when:** `from src.config import RANDOM_SEED` works from any script.

## Stage 2: Data loading

**Goal:** load raw SomaScan + clinical data, filter to the analysis cohort, return `(X, y, protein_ids, patient_ids)`.

**File:** `src/data_loading.py`

**Function signature:**

```python
def load_cohort(data_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Return X (n_patients x n_proteins), y (binary ARDS label), protein_ids, patient_ids."""
```

**Filtering rules:**

- Keep patients with sepsis.
- Keep patients with ARDS classified as moderate or severe → `y = 1`.
- Keep patients with sepsis but no ARDS → `y = 0`.
- Drop patients with mild ARDS.
- Drop patients with missing ARDS label.

**Script:** `scripts/00_prepare_data.py`

- Calls `load_cohort()`.
- Saves `data/processed/cohort.parquet` with columns `[patient_id, <protein_1>, ..., <protein_p>, y]`.
- Saves `data/processed/protein_ids.json` with the list of protein IDs in order.
- Logs: n_patients, n_proteins, class balance, any dropped patients and reason.

**Note:** do not transform protein values. SomaScan output is assumed already normalised upstream. Verify with data provider before running on real data; log the assumed normalisation state.

**Done when:** `cohort.parquet` exists and class balance matches expected (~20% ARDS).

## Stage 3: Split generation

**Goal:** generate K stratified train/test splits once, save to disk, reuse everywhere.

**File:** `src/splits.py`

**Function signature:**

```python
def generate_splits(y: np.ndarray, k: int, test_size: float, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_idx, test_idx) tuples using StratifiedShuffleSplit."""
```

**Implementation:** wrap `sklearn.model_selection.StratifiedShuffleSplit` with the given parameters. Return precomputed indices as a list of tuples.

**Script:** `scripts/01_generate_splits.py`

- Loads `cohort.parquet`.
- Generates `K_SPLITS = 50` splits.
- Saves to `data/processed/splits.pkl` as a list of `(train_idx, test_idx)` numpy arrays.
- Also saves `data/processed/splits_meta.json` with seed, K, test_size, n_ards_per_test_fold (list) for sanity checking.

**Expensive-method subset:** expensive methods use the first `K_SPLITS_EXPENSIVE = 10` splits from the same file. Do not regenerate.

**Done when:** `splits.pkl` exists, class balance in test folds is consistent (all ~20% ARDS).

## Stage 4: Selection method interface + t-test + random baseline

**Goal:** establish the plugin architecture for selection methods, implement the two baselines.

**File:** `src/selection/base.py`

```python
from abc import ABC, abstractmethod
import numpy as np

class SelectionMethod(ABC):
    name: str  # unique identifier, used in paths and logs

    @abstractmethod
    def select(self, X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (ranked_indices, scores) where ranked_indices[0] is most important.
        scores is aligned with ranked_indices, same length.
        Length equals the number of proteins the method considers selected at its native cutoff.
        For ranking-only methods (no native cutoff), return all proteins ranked.
        """
        pass
```

**File:** `src/selection/ttest.py`

- Welch's t-test per protein, Benjamini-Hochberg FDR correction at q < 0.05.
- Return proteins with q < 0.05, ranked by ascending q-value. Also return the q-values as scores.
- If no protein passes FDR, return empty arrays.

**File:** `src/selection/random.py`

- Return a random subset of proteins, size specified by constructor argument (default: match median of other methods; for initial runs, set to 30).
- Scores are uniform random in [0, 1].
- Use the per-split seed so different splits give different random selections, but a given split is reproducible.

**File:** `scripts/02_run_selection.py`

- Args: `--method <name>`, optional `--n-splits <int>` to override K (for expensive methods).
- Loads cohort + splits.
- For each split: extract X_train, y_train, run method, save (ranked_indices, scores) for that split.
- Output: `results/selection/<method>/selections.parquet` with columns `[split_id, rank, protein_idx, protein_id, score]`. Long format, one row per (split, protein) for every protein the method selected.
- Output: `results/selection/<method>/meta.json` with method name, n_splits, timing, any method-specific parameters.
- Logs: number of proteins selected per split (min, median, max), total runtime.

**Method registry:** in `scripts/02_run_selection.py`, map `--method` string to class via a dict. Adding a new method means importing it and adding one line to the registry.

**Done when:** running `python scripts/02_run_selection.py --method ttest` produces a valid selections.parquet with >0 rows for most splits.

## Stage 5: Stability computation

**Goal:** compute selection frequency per protein, derive stable sets.

**File:** extend `src/selection/base.py` or new module `src/stability.py`.

**Function:**

```python
def compute_stability(selections: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    """Input: selections long-format df. Output: df with [protein_idx, protein_id, frequency, mean_rank]."""
```

**Stable set construction:**

```python
def stable_set(stability_df: pd.DataFrame, pi: float, min_size: int, topk_fallback: int) -> list[str]:
    """Return protein_ids with frequency >= pi. If fewer than min_size, fall back to top-topk_fallback by frequency."""
```

**Integrate into `scripts/02_run_selection.py`:**

- After per-split selections are saved, compute and save:
  - `results/selection/<method>/stability.parquet` with all proteins ever selected, their frequency, mean rank when selected.
  - `results/selection/<method>/stable_set.json` with the derived stable set (primary: π = 0.3, fallback: top-50 by frequency).

**Done when:** `stability.parquet` exists and selection frequency curve can be plotted.

## Stage 6: Classifier validation

**Goal:** train LR, RF, XGB per split using the method's top-k selection; evaluate on test fold.

**File:** `src/classifiers.py`

- Three constructor functions returning sklearn-compatible estimators with fixed configs from `config.py`.
- `build_logreg()`, `build_rf()`, `build_xgb()`.
- XGB: `scale_pos_weight = n_neg / n_pos` computed per training fold.

**File:** `src/metrics.py`

- `compute_auc(y_true, y_score) -> float`
- `compute_aupr(y_true, y_score) -> float`
- Handle edge cases (test fold with only one class → return NaN, log warning).

**File:** `scripts/03_run_classifiers.py`

- Args: `--method <name>`.
- Loads cohort + splits + method's per-split selections.
- For each split:
  - For each k in `TOPK_VALUES`:
    - Take top-k protein indices for this split from selections.parquet.
    - For each classifier (LR, RF, XGB):
      - Train on X_train[:, topk_idx], y_train.
      - Predict on X_test[:, topk_idx], compute AUC and AUC-PR.
- Output: `results/classifier/<method>/scores.parquet` with columns `[split_id, classifier, k, auc, aupr, n_features_actual]`.
  - `n_features_actual` may be less than k if the method selected fewer than k proteins in that split.

**Also compute:** native-cutoff classifier performance (k = number of proteins at method's native threshold for that split). Save as separate rows with `k = "native"`.

**Done when:** scores.parquet contains entries for all (split, classifier, k) combinations.

## Stage 7: Enrichment validation

**Goal:** run ORA on the stable set, compute BES and permutation null.

**File:** `src/enrichment.py`

**Functions:**

```python
def run_ora(gene_list: list[str], background: list[str], library: str) -> pd.DataFrame:
    """Call Enrichr via gseapy, return term table with columns including term, q_value, genes, term_size."""

def compute_bes(term_df: pd.DataFrame, gene_list: list[str], c: float, tau: float, q_threshold: float) -> dict:
    """Compute BES and its components. Return dict with raw BES, per-term contributions, diagnostic info."""

def permutation_null(background: list[str], gene_list_size: int, library: str,
                     b_perm: int, c: float, tau: float, q_threshold: float,
                     seed: int) -> np.ndarray:
    """Return array of BES values under random gene list sampling."""
```

**Implementation notes:**

- `run_ora`: use `gseapy.enrich()` with background set. Parse output into a uniform DataFrame.
- `compute_bes`: implement the formula from the thesis:
  - Filter to terms with q < q_threshold.
  - For each term: compute s_i = min(-log10(q), c), w_i = u_i * a_i * g_i.
  - u_i from pairwise Jaccard with τ threshold.
  - a_i = 1 / log2(|T_i| + 1).
  - g_i = |T_i ∩ G| / |G|.
  - Sum w_i * s_i over terms.
- `permutation_null`: sample random gene lists from background, run ORA on each, compute BES. Cache results by (gene_list_size, library) to share across methods.

**File:** `scripts/04_run_enrichment.py`

- Args: `--method <name>`, optional `--skip-null` for quick runs.
- Loads stable set for method.
- For each library in `ENRICHMENT_LIBRARIES`:
  - Run ORA, save term table to `results/enrichment/<method>/<library>/terms.parquet`.
  - Compute BES, save components.
  - Compute or load cached permutation null.
  - Compute z-score and empirical p-value.
- Output: `results/enrichment/<method>/summary.parquet` with columns `[library, gene_list_size, bes_raw, bes_z, bes_p_emp, n_significant_terms]`.

**Permutation null cache:**

- Directory: `results/enrichment/_null_cache/<library>/`.
- Filename: `size_<N>_seed_<seed>.npz` with the array of BES values under null.
- Before computing, check if cache exists. Skip if so.
- Reason: expensive (1000 ORA calls per (size, library)), reusable across methods at the same stable-set size.

**Done when:** summary.parquet exists for at least one method and at least one library.

## Stage 8: Simulation validation

**Goal:** generate synthetic data with planted signals; evaluate each method's recall and FDR per signal type.

**File:** `src/simulation.py`

**Functions:**

```python
def generate_simulated_dataset(n: int, p: int, class_prevalence: float,
                               signal_specs: list[dict], seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return X, y, ground_truth. ground_truth maps signal type -> list of feature indices."""
```

**Signal types to plant (see methodology):**

- **Linear:** X_j | Y ~ N(mu_y, 1), delta_mu determined by effect size.
- **Saturation:** nonlinear monotonic, class-dependent via tanh transform.
- **U-shape:** X_j | Y=1 ~ mixture of N(+a, σ²) + N(-a, σ²), X_j | Y=0 ~ N(0, σ'²) with matched mean.
- **Threshold:** X_j informative only above cutoff.
- **XOR pair:** (X_j, X_k) jointly determine Y via sign interaction; neither has marginal association.

Each signal type planted at `SIM_SIGNALS_PER_TYPE` = 3 features per effect size level. Remaining features are iid N(0, 1) noise.

**File:** `scripts/05_run_simulation.py`

- Args: `--method <name>`, optional `--n-repeats <int>`.
- For each repeat r in 1..R:
  - Generate simulated dataset with seed = global_seed + r.
  - Run the selection method (same interface as real-data selection).
  - For each k in TOPK_VALUES:
    - Compute recall per signal type: fraction of planted signals of that type in top-k.
    - Compute FDR: fraction of top-k that is noise.
- Output: `results/simulation/<method>/results.parquet` with columns `[repeat, signal_type, effect_size, k, recall, fdr]`.

**Done when:** results.parquet exists for at least one method with all signal types represented.

## Stage 9: Cross-method aggregation

**Goal:** produce the comparison tables and figures used in the cross-method comparison section of the thesis.

**File:** `scripts/06_aggregate_results.py`

**Outputs:**

- `results/comparison/classifier_summary.parquet`: mean ± SD AUC / AUC-PR per (method, classifier, k).
- `results/comparison/stability_summary.parquet`: per method: size of stable set at π = 0.3, top frequency, number ≥ 0.5, number ≥ 0.3.
- `results/comparison/enrichment_summary.parquet`: per (method, library): BES, z, p_emp, n_significant_terms.
- `results/comparison/simulation_summary.parquet`: mean recall and FDR per (method, signal_type, effect_size, k).
- `results/comparison/protein_overlap.parquet`: pairwise Jaccard of stable sets across methods.

**Figures (saved to results/comparison/figures/):**

- Selection frequency curves per method (rank vs frequency, log-y).
- Classifier AUC per method × classifier at top-k = 50 (bar chart with error bars).
- Simulation recall heatmap: methods × signal types.
- BES z-score per method per library (bar chart).

Do figures in a notebook (`notebooks/results_figures.ipynb`) that reads from the parquet summaries. Do not embed plotting in the aggregation script.

**Done when:** all summary parquets exist and the figures notebook runs end to end.

## Stage 10: Testing (minimal)

**Goal:** sanity checks on the critical functions, not full coverage.

**File:** `tests/test_selection.py`

- Test that t-test on data with known signal (e.g., a single linearly associated feature) selects that feature.
- Test that random selection returns the correct number of features.
- Test that the base class raises if `select` is not implemented.

**File:** `tests/test_simulation.py`

- Test that generated XOR pair has no marginal association (both features individually uncorrelated with y).
- Test that U-shape features have same class means but different variances.
- Test that ground truth indices match actual planted features.

**File:** `tests/test_bes.py`

- Test BES on a trivial case (single significant term, no redundancy): value matches hand computation.
- Test that BES is zero when no terms pass q_threshold.

**Done when:** `pytest tests/` passes.

## Implementation order

Do stages in this order. Do not start stage N+1 before stage N is done end to end.

1. **Stage 0** (scaffolding) → commit.
2. **Stage 1** (config) → commit.
3. **Stage 2** (data loading) → verify on real data, commit.
4. **Stage 3** (splits) → commit.
5. **Stage 4** (t-test + random baseline) → verify stability numbers match the 72%/30% pattern observed → commit.
6. **Stage 5** (stability) → commit.
7. **Stage 6** (classifiers) → run on t-test, confirm AUC is plausible (>0.5, <1.0) → commit.
8. **Stage 7** (enrichment) → run on t-test, confirm BES and null computation → commit.
9. **Stage 8** (simulation) → run on t-test, confirm it detects linear signals well and non-linear poorly → commit. This is a critical sanity check.
10. **Stage 9** (aggregation) → produce first draft of comparison outputs for t-test + random only → commit.
11. **Stage 10** (tests) → commit.

**Only then** start adding new selection methods (MI, dCor, HSIC, interaction methods). Each new method is a single file in `src/selection/`, one line added to the method registry, then `02`–`05` scripts rerun for that method. The rest of the pipeline is unchanged.

## Things not to do

- Do not mix exploration and pipeline code. Notebooks read from `results/`, they never write to `results/`.
- Do not recompute things that are already cached. Check for existence of output files before running expensive stages.
- Do not rely on a single global seed for everything. Per-split seeds are derived from split_id.
- Do not hand-edit result files.
- Do not store method-specific hyperparameters in `config.py` beyond the shared ones. Method-specific parameters live in the method's own file with defaults that can be overridden via constructor arguments.
- Do not skip the simulation validation. It is the ground-truthed check that the methods do what they claim.
- Do not tune classifier hyperparameters. This is a deliberate design choice documented in the methodology.

## Reasonable first-day deliverable

After stages 0–3: you have an environment, a config, a filtered cohort, and cached splits. Roughly half a day of work.

After stages 4–6 (t-test only): you have per-split selections, stability curve, classifier AUCs for t-test. This is already enough to draft the t-test results subsection of the thesis. One to two days depending on familiarity with the tooling.

After stages 7–8: enrichment and simulation for t-test. Enrichment is slow because of the permutation null (plan for overnight runs); simulation is fast.

Everything after is adding new selection methods one at a time. Each new method: ~half a day for implementation, overnight for full pipeline run.