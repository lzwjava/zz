#!/usr/bin/env bash
# ============================================================
# prepare_data.sh — download + convert all datasets for nanochat
# ============================================================
# Step 1: Download github-code parquets (38 shards = 11 GB, of 1126 total)
# Step 2: Convert to nanochat format (content col → text col)
# Step 3: Download fineweb-edu (~10 shards = 16 GB, 4.3B tokens)
# Step 4: Convert sec-edgar to nanochat format (extract text col)
# Step 5: Merge everything into a single nanochat data dir
#
# Usage:
#   bash prepare_data.sh                         # full run
#   bash prepare_data.sh --skip-fineweb-dl        # skip fineweb download (use cached)
# ============================================================
set -euo pipefail

ZZ_DIR="/mnt/data/zz"
NANOCHAT_DIR="/mnt/data/nanochat"
export NANOCHAT_DATA_DIR="$ZZ_DIR/datasets/nanochat-mixed"

SKIP_FW="${1:-}"

echo "========================================"
echo "Step 1/5: Convert github-code parquets"
echo "========================================"
cd "$NANOCHAT_DIR" && source .venv/bin/activate
python3 "$ZZ_DIR/scripts/extract/convert_github_code_for_nanochat.py"

echo ""
echo "========================================"
echo "Step 2/5: Download fineweb-edu (9 shards, ~4.3B tokens)"
echo "========================================"
if [ "$SKIP_FW" != "--skip-fineweb-dl" ]; then
  cd "$ZZ_DIR"
  python3.11 scripts/download/plan_and_download_fineweb.py \
    --target-tokens 5B \
    --dataset fineweb-edu \
    --output-dir "$ZZ_DIR/datasets/fineweb-edu"
else
  echo "  --skip-fineweb-dl: using cached fineweb-edu parquets"
fi

# Validate fineweb parquets (some may be corrupt from partial download)
echo "  Validating fineweb parquets..."
cd "$NANOCHAT_DIR" && source .venv/bin/activate
python3 -c "
import pyarrow.parquet as pq, glob, os
fw_dir = '$ZZ_DIR/datasets/fineweb-edu'
ok = bad = 0
for f in sorted(glob.glob(fw_dir + '/*.parquet')):
    try:
        pq.ParquetFile(f)
        ok += 1
    except:
        print(f'  CORRUPT: removing {os.path.basename(f)}')
        os.remove(f)
        bad += 1
print(f'  {ok} good, {bad} removed')
"

echo ""
echo "========================================"
echo "Step 3/5: Convert sec-edgar parquets"
echo "========================================"
cd "$NANOCHAT_DIR" && source .venv/bin/activate
python3 "$ZZ_DIR/scripts/extract/convert_sec_edgar_for_nanochat.py"

echo ""
echo "========================================"
echo "Step 4/5: Merge into $NANOCHAT_DATA_DIR"
echo "========================================"
mkdir -p "$NANOCHAT_DATA_DIR"
rm -f "$NANOCHAT_DATA_DIR"/*.parquet

# github-code (train)
for f in "$ZZ_DIR/datasets/github-code-nanochat/train_"*.parquet; do
  ln -sf "$(realpath "$f")" "$NANOCHAT_DATA_DIR/code_$(basename "$f")"
done

# sec-edgar (train)
for f in "$ZZ_DIR/datasets/sec-edgar-nanochat/"*.parquet; do
  ln -sf "$(realpath "$f")" "$NANOCHAT_DATA_DIR/sec_$(basename "$f")"
done

# fineweb-edu (first 8 = train, last = val)
fw_files=($(ls "$ZZ_DIR/datasets/fineweb-edu/"*.parquet 2>/dev/null))
fw_count=${#fw_files[@]}
if [ "$fw_count" -gt 0 ]; then
  # First (fw_count-1) go to train
  for ((i=0; i<fw_count-1; i++)); do
    f="${fw_files[$i]}"
    num=$(basename "$f" | sed 's/.*train-0*\([0-9]*\)-of.*/\1/')
    ln -sf "$(realpath "$f")" "$NANOCHAT_DATA_DIR/fw_$(printf '%05d' "$num").parquet"
  done
  # Last shard = validation
  last="${fw_files[$fw_count-1]}"
  ln -sf "$(realpath "$last")" "$NANOCHAT_DATA_DIR/zz_val.parquet"
fi

echo "  Data dir: $NANOCHAT_DATA_DIR"
echo "  Total files: $(ls "$NANOCHAT_DATA_DIR"/*.parquet | wc -l)"
echo "  Train files: $(ls "$NANOCHAT_DATA_DIR"/*.parquet | grep -v zz_val | wc -l)"
echo "  Val file: zz_val.parquet ($(du -h "$NANOCHAT_DATA_DIR/zz_val.parquet" | cut -f1))"

echo ""
echo "========================================"
echo "Step 5/5: Train tokenizer on mixed data"
echo "========================================"
cd "$NANOCHAT_DIR" && source .venv/bin/activate
python -m scripts.tok_train --max-chars 2000000000 --vocab-size 32768

echo ""
echo "=== DONE ==="
echo "Ready to train:"
echo "  bash $ZZ_DIR/fineweb-code-sec-gpt.sh              # full 50k run"
echo "  bash $ZZ_DIR/fineweb-code-sec-gpt.sh --smoke       # smoke test"