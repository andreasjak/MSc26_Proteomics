# MSc26 Proteomics

Pipeline project for identifying non-linear protein regulation and higher-order interactions in ARDS from SomaScan proteomics data.

## Setup

Create and activate the Conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate proteomicsEnv
```

If dependencies change:

```bash
conda env update -f environment.yml --prune
```

## Stage 0 Scaffold

This repository follows the staged implementation plan in `IMPLEMENTATION_GUIDE.md`.
Stage 0 establishes the project scaffold used by later stages:

```text
src/
	config.py
	data_loading.py
	splits.py
	selection/
		__init__.py
		base.py
		ttest.py
		random.py
	classifiers.py
	enrichment.py
	simulation.py
	metrics.py
	utils.py

scripts/
	00_prepare_data.py
	01_generate_splits.py
	02_run_selection.py
	03_run_classifiers.py
	04_run_enrichment.py
	05_run_simulation.py
	06_aggregate_results.py
```

Run all scripts from the repository root:

```bash
python ./scripts/<script_name>.py -h
```
