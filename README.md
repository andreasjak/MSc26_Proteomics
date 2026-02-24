# MSc26 Proteomics

Research project investigating proteomics data analysis

## Project Structure

```
MSc26_Proteomics/
├── logs/                          # Logs from runs (not tracked in git)
├── results/                       # Results (not tracked in git)
├── data/                          # Data files (not tracked in git)
│   ├── raw/                       # Raw, unprocessed data
│   └── processed/                 # Processed data from preprocessing scripts
├── notebooks/                     # Jupyter notebooks for exploration and analysis
├── scripts/                       # Standalone scripts and pipelines
├── src/                           # Source code and reusable modules
│   ├── styles/
│   └── core/
├── tests/                         # Unit tests
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Running scripts

- All scripts now use **CLI arguments** (`argparse`) only.
- Open a terminal in the project root (`MSc26_Proteomics`) and run commands from there.

### General command format

```powershell
python .\scripts\<script_name>.py --arg1 value1 --arg2 value2
```

### Example

```powershell
python .\scripts\ttest.py --data-path .\data\processed\seen.csv --save-results --k 25
```

This example runs `ttest.py` on `seen.csv`, saves outputs (instead of interactive plotting), and writes a top-25 feature file (`selected_features_k25.csv`) along with the t-test results.

### Help with scripts
For help, discription and usage of script run:

```powershell
python .\scripts\<script_name>.py -h
```
or 
```powershell
python .\scripts\<script_name>.py --help
```

## Environment setup
### Create the environment
From the project root (where `environment.yml` is located):

```bash
conda env create -f environment.yml
```

### Activate the environment
```bash
conda activate proteomicsEnv
```

### Update the environment 
If `environment.yml` changes:
```bash
conda env update -f environment.yml --prune
```
