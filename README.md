# ZZ

Dataset processing and training utilities for machine learning projects.

## Setup

```bash
pip install -r requirements.txt
```

## Directory Structure

```
scripts/
  download/   # Dataset download scripts
  extract/    # Data extraction scripts
  analysis/   # Training analysis and evaluation
logs/         # Training logs and outputs
```

## Usage

### Download Datasets

```bash
# Download FineWeb dataset
python scripts/download/download_fineweb.py --limit 1000 --output output.txt

# Download with wget scripts
bash scripts/download/wget_fineweb_1.sh
```

### Extract Data

```bash
# Extract from parquet files
python scripts/extract/extract_parquet.py

# Extract FineWeb data
python scripts/extract/extract_fineweb.py
```

### Analysis

```bash
# Calculate training duration
python scripts/analysis/calculate_duration.py

# Evaluate training metrics
python scripts/analysis/evaluate.py --file logs/train_log_openweb.txt
```
