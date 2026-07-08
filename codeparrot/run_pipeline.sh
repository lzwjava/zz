#!/usr/bin/env bash
set -euo pipefail

# Full pipeline: download codeparrot-clean (Python-only) -> convert -> tokenize
# ~12.7 GB download, ~50 GB uncompressed, 5.3M Python files

DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Step 1: Download all 53 train shards + 1 valid shard ==="
python3.11 "$DIR/download.py"

echo ""
echo "=== Step 2: Convert .json.gz NDJSON -> text-column parquet ==="
python3.11 "$DIR/convert.py"

echo ""
echo "=== Step 3: Tokenize -> .npy shards for nanoGPT/nanochat ==="
python3.11 "$DIR/tokenize.py" \
    --input-dir /mnt/data/zz/datasets/codeparrot-clean-nanochat \
    --output-dir /mnt/data/zz/datasets/codeparrot-clean-tok

echo ""
echo "=== Done ==="
echo "Tokenized shards: /mnt/data/zz/datasets/codeparrot-clean-tok/"