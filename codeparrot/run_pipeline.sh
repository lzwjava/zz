#!/usr/bin/env bash
set -euo pipefail

# Full pipeline: download codeparrot-clean (Python-only) → convert → tokenize
# ~12.7 GB download, ~50 GB uncompressed, 5.3M Python files

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Step 1: Download all 53 train shards + 1 valid shard ==="
python3.11 "$ROOT/scripts/download/download_codeparrot_clean.py"

echo ""
echo "=== Step 2: Convert .json.gz NDJSON → text-column parquet ==="
python3.11 "$ROOT/scripts/extract/convert_codeparrot_for_nanochat.py"

echo ""
echo "=== Step 3: Tokenize → .npy shards for nanoGPT/nanochat ==="
python3.11 "$ROOT/scripts/extract/tokenize_github_code.py \
    --input-dir /mnt/data/zz/datasets/codeparrot-clean-nanochat \
    --output-dir /mnt/data/zz/datasets/codeparrot-clean-tok"

echo ""
echo "=== Done ==="
echo "Tokenized shards: /mnt/data/zz/datasets/codeparrot-clean-tok/"