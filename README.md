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

## Pending Features

- **RAG (Retrieval-Augmented Generation)** — local knowledge base Q&A using vector search over your own documents
- **Knowledge Base Management** — ingest PDFs, text files, and other documents into a vector store (e.g. FAISS)
- **LLM Integration** — connect local or API-based models for document-grounded answers
- **Agent Support** — tool-calling agents that can query knowledge bases and external APIs
- **Web UI** — chat interface for interacting with the knowledge base

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
