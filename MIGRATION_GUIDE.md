# Migration Guide: Stability-Based → Full-Data Selection Pipeline

## Summary of change

The current pipeline derives the final protein list from stability selection across K resampling splits. The new design separates method evaluation (resampling + classifier) from the final protein list (selection on full data). Stability becomes an optional diagnostic, not the selection mechanism.

**Before:** resampling splits → per-split selection → stability frequency → stable set → enrichment on stable set.
**After:** resampling splits → per-split selection → classifier validation (unchanged). Separately: full-data selection → final protein list → enrichment on final list. Stability computed as diagnostic only.

## Changes by file

### 1. `scripts/02_run_selection.py`

**Add a `--full-data` flag.**

When `--full-data` is passed:
- Skip loading splits entirely.
- Load the full cohort X, y.
- Run the selection method once on the full X, y.
- Save output to `results/selection/<method>/full_data_ranking.parquet` with the same columns as `selections.parquet`: `[rank, protein_idx, protein_id, score, significant]`. No `split_id` column.
- Save `results/selection/<method>/full_data_meta.json` with method name, timing, parameters, n_samples, n_ards.

When `--full-data` is NOT passed: behaviour is unchanged (per-split selection as before).

**Typical usage:**
```bash
# Method evaluation (unchanged)
python scripts/02_run_selection.py --method ttest

# Final protein list (new)
python scripts/02_run_selection.py --method ttest --full-data
```

### 2. `scripts/04_run_enrichment.py`

**Change input source.**

Currently reads from: `results/selection/<method>/stable_set.json`
Now reads from: `results/selection/<method>/full_data_ranking.parquet`

Derive the gene list from `full_data_ranking.parquet`:
- Primary: proteins where `significant == True`.
- If fewer than `STABILITY_MIN_SIZE` proteins are significant, fall back to top-`STABILITY_MIN_SIZE` by rank.
- Save the derived gene list to `results/enrichment/<method>/gene_list.json` for traceability.

Everything else in the enrichment pipeline (ORA, BES, permutation null) is unchanged. Only the input gene list source changes.

### 3. `src/stability.py`

**No functional changes.** Keep all functions as they are.

### 4. `scripts/02_run_selection.py` (stability section)

**Keep stability computation but reframe outputs.**

The stability parquets (`stability_significant.parquet`, `stability_topk.parquet`) are still computed after per-split selection. They are now diagnostic outputs, not inputs to downstream stages.

**Remove or rename `stable_set.json`:** this file is no longer used by enrichment. Either:
- Stop generating it (simplest).
- Rename to `stability_diagnostic.json` and keep it as a diagnostic artifact that is not consumed by any downstream script.

### 5. `scripts/06_aggregate_results.py`

**Update stability summary.**

Currently: reports stable set size, pi used, fallback method.
Now: reports selection frequency statistics as method diagnostics:
- Top protein frequency.
- Number of proteins with frequency >= 0.3, >= 0.5.
- Mean pairwise Jaccard of top-50 across splits.

These are reported alongside classifier and enrichment results but are not used to derive any final protein list.

**Add full-data selection summary.**

New table in `results/comparison/`:
- `full_data_summary.parquet` with columns `[method, n_significant, top_protein, top_score]`.
- Overlap matrix: pairwise Jaccard of full-data significant sets across methods.

### 6. `scripts/03_run_classifiers.py`

**No changes.** Classifier validation still runs per split using per-split selections from `selections.parquet`. This is the method evaluation loop and is unaffected by the full-data change.

### 7. `scripts/05_run_simulation.py`

**No changes.** Simulation is independent of real-data pipeline.

### 8. `src/config.py`

**No changes required.** `STABILITY_PI_PRIMARY` and `STABILITY_MIN_SIZE` can stay — they are still used for diagnostics. Optionally add a comment marking them as diagnostic-only.

## Execution order after migration

```bash
# 1. Per-split selection (method evaluation, unchanged)
python scripts/02_run_selection.py --method ttest

# 2. Full-data selection (new step)
python scripts/02_run_selection.py --method ttest --full-data

# 3. Classifier validation (unchanged, reads per-split selections)
python scripts/03_run_classifiers.py --method ttest

# 4. Enrichment (now reads full-data ranking instead of stable set)
python scripts/04_run_enrichment.py --method ttest

# 5. Simulation (unchanged)
python scripts/05_run_simulation.py --method ttest

# 6. Aggregation (minor updates)
python scripts/06_aggregate_results.py
```

## Files affected summary

| File | Change type |
|---|---|
| `scripts/02_run_selection.py` | Add `--full-data` flag and full-data execution path |
| `scripts/04_run_enrichment.py` | Change input from `stable_set.json` to `full_data_ranking.parquet` |
| `scripts/06_aggregate_results.py` | Add full-data summary, reframe stability as diagnostic |
| `scripts/03_run_classifiers.py` | No change |
| `scripts/05_run_simulation.py` | No change |
| `src/stability.py` | No change |
| `src/config.py` | No change (optional comment) |
| `src/enrichment.py` | No change (input gene list is passed in, source is caller's concern) |

## New output files

| File | Description |
|---|---|
| `results/selection/<method>/full_data_ranking.parquet` | Full ranking of all proteins on full cohort |
| `results/selection/<method>/full_data_meta.json` | Metadata for full-data run |
| `results/enrichment/<method>/gene_list.json` | Gene list derived from full-data ranking, used for ORA |
| `results/comparison/full_data_summary.parquet` | Cross-method summary of full-data selections |