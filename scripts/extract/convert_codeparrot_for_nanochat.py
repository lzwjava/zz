#!/usr/bin/env python3
"""Convert codeparrot-clean .json.gz (NDJSON) → nanochat format (text col parquet).

Reads each gzipped shard line by line, extracts the `content` field,
and writes a parquet file with a single `text` column.
Memory-efficient: processes line-by-line, writes in batches.

Usage:
    python3.11 scripts/extract/convert_codeparrot_for_nanochat.py

Source: /mnt/data/zz/datasets/codeparrot-clean/file-*.json.gz
Output: /mnt/data/zz/datasets/codeparrot-clean-nanochat/train_*.parquet
"""

import gzip
import json
import glob
import os
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

SRC_DIR = "/mnt/data/zz/datasets/codeparrot-clean"
DST_DIR = "/mnt/data/zz/datasets/codeparrot-clean-nanochat"

BATCH_SIZE = 50000  # rows per parquet row-group


def get_shard_number(filename):
    """Extract shard number from file-000000000042.json.gz"""
    stem = filename.replace("file-", "").replace(".json.gz", "")
    return int(stem)


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    sources = sorted(glob.glob(os.path.join(SRC_DIR, "file-*.json.gz")))
    if not sources:
        print(f"ERROR: No .json.gz files in {SRC_DIR}")
        print("Run download_codeparrot_clean.py first.")
        sys.exit(1)

    total_size = sum(os.path.getsize(f) for f in sources)
    print(f"Found {len(sources)} gzip shards ({total_size / 1024**3:.1f} GB compressed)")

    t0 = time.time()
    total_rows = 0
    skipped_shards = 0

    for si, src in enumerate(sources, 1):
        base = os.path.basename(src)
        shard_num = get_shard_number(base)

        # First 53 files = train, file 54 = valid set from codeparrot-clean-valid? 
        # Actually file 54 was on the valid repo. Let's just number by shard.
        split = "val" if shard_num == 54 else "train"
        dst = os.path.join(DST_DIR, f"{split}_{shard_num:04d}.parquet")

        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            sz = os.path.getsize(dst)
            print(f"  [{si}/{len(sources)}] cached {os.path.basename(dst)} ({sz/1e6:.0f} MB)")
            skipped_shards += 1
            rows = pq.ParquetFile(dst).metadata.num_rows
            total_rows += rows
            continue

        print(f"  [{si}/{len(sources)}] {base}...", end="", flush=True)

        try:
            texts = []
            writer = None
            n_parsed = 0
            n_skipped = 0

            with gzip.open(src, "rt", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        content = obj.get("content", "")
                        if content and len(content) >= 10:
                            texts.append(content)
                            n_parsed += 1
                        else:
                            n_skipped += 1
                    except json.JSONDecodeError:
                        n_skipped += 1
                        continue

                    # Flush batch
                    if len(texts) >= BATCH_SIZE:
                        batch = pa.table({"text": pa.array(texts, type=pa.string())})
                        if writer is None:
                            writer = pq.ParquetWriter(dst, batch.schema)
                        writer.write_table(batch)
                        texts = []

            # Flush remaining
            if texts:
                batch = pa.table({"text": pa.array(texts, type=pa.string())})
                if writer is None:
                    writer = pq.ParquetWriter(dst, batch.schema)
                writer.write_table(batch)

            if writer:
                writer.close()

            sz = os.path.getsize(dst) if os.path.exists(dst) else 0
            total_rows += n_parsed
            elapsed = time.time() - t0
            print(f" {n_parsed:,} rows, {sz/1e6:.0f} MB "
                  f"({n_skipped} skipped) | "
                  f"{elapsed/60:.1f} min")

        except Exception as e:
            print(f" FAILED: {e}")
            if os.path.exists(dst):
                os.remove(dst)
            sys.exit(1)

    elapsed = time.time() - t0
    all_dst = sorted(glob.glob(os.path.join(DST_DIR, "*.parquet")))
    total_bytes = sum(os.path.getsize(f) for f in all_dst)
    print(f"\nDone. {len(all_dst)} shards, {total_rows:,} rows, "
          f"{total_bytes/1e9:.2f} GB, {elapsed/60:.1f} min")
    print(f"\nNext step — tokenize for nanoGPT/nanochat:")
    print(f"  python3.11 scripts/extract/tokenize_github_code.py \\")
    print(f"    --input-dir {DST_DIR} \\")
    print(f"    --output-dir /mnt/data/zz/datasets/codeparrot-clean-tok")


if __name__ == "__main__":
    main()