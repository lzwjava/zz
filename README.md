# ZZ

Dataset processing, tokenization, and model inference utilities for ML training pipelines.

## Setup

```bash
pip install -r requirements.txt
```

## Directory Structure

```
scripts/
  download/     # Dataset download scripts (FineWeb, Wikimedia, HF mirrors)
  extract/      # Data extraction, tokenization, and renaming
  analysis/     # Training duration and metric evaluation
  deepseek/     # LLM inference scripts (DeepSeek-V2-Lite)
logs/           # Training logs and outputs
datasets/       # Downloaded dataset storage
```

## Usage

### Download Datasets

```bash
# Download FineWeb dataset (simple)
python scripts/download/download_fineweb.py --limit 1000 --output output.txt

# Plan and download shards to hit a token budget
python scripts/download/plan_and_download_fineweb.py --tokens 10B --output /mnt/data/zz/datasets/fineweb

# Download ~100B tokens for GPT-3 ablation (resumable, progress.json)
python scripts/download/plan_and_download_fineweb_gpt3.py --output /mnt/data/zz/datasets/fineweb-edu

# Download and extract in one step (pyarrow iter_batches, memory-safe)
python scripts/download/download_and_extract.py

# Download via wget scripts (hf-mirror.com for China access)
bash scripts/download/wget_fineweb_mirror_1.sh
bash scripts/download/wget_fineweb_mirror_2_5.sh
bash scripts/download/wget_fineweb_mirror_11_20.sh

# Wikimedia dumps
bash scripts/download/wget_wikimedia_1.sh
bash scripts/download/wget_wikimedia_4.sh
bash scripts/download/wget_wikimedia_4_accum.sh
bash scripts/download/wget_wikimedia_5.sh
```

### Extract and Tokenize

```bash
# Extract from parquet files
python scripts/extract/extract_parquet.py

# Extract FineWeb data
python scripts/extract/extract_fineweb.py

# Extract GPT-3 ablation shards -> single train_fineweb.txt
python scripts/extract/extract_fineweb_gpt3.py

# Tokenize GPT-3 ablation shards -> uint16 .npy shards (GPT-2 BPE)
python scripts/extract/tokenize_fineweb_gpt3.py

# Rename downloaded parquet files (strip ?download=true suffix)
python scripts/extract/rename_fineweb.py

# Extract Wikipedia dumps
python scripts/extract/extract_wiki.py
python scripts/extract/extract_wiki_corpus.py
```

### Analysis

```bash
# Calculate training duration
python scripts/analysis/calculate_duration.py

# Evaluate training metrics
python scripts/analysis/evaluate.py --file logs/train_log_openweb.txt
```

### Model Inference

```bash
# DeepSeek-V2-Lite-Chat (4-bit quantized, fits 12GB VRAM)
python scripts/deepseek/run_lite.py                       # interactive chat
python scripts/deepseek/run_lite.py -p "Your prompt"      # single prompt
```
