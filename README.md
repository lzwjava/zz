# ZZ

Dataset processing and training utilities for machine learning projects.

## Setup

```bash
pip install -r requirements.txt
```

## Scripts

### Dataset Download & Extraction

- `download_fineweb.py` - Download FineWeb dataset with streaming support
- `download_and_extract.py` - Download and extract FineWeb using pyarrow
- `extract_parquet.py` - Extract data from parquet files

### Training Utilities

- `train/calculate_duration.py` - Calculate training duration from logs
- `train/evaluate.py` - Model evaluation
- `train/extract_wiki.py` - Extract Wikipedia corpus
- `train/extract_wiki_corpus.py` - Process Wikipedia data
- `train/extract_fineweb.py` - Extract FineWeb data
- `train/rename_fineweb.py` - Rename FineWeb files

## Usage

Download FineWeb dataset:
```bash
python download_fineweb.py --limit 1000 --output output.txt
```

Calculate training duration:
```bash
python train/calculate_duration.py
```
