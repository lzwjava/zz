#!/usr/bin/env python3
"""Convert github-code parquets (content col) → nanochat format (text col).

Reads each source shard row-group by row-group, extracts the `content`
column, and writes a new parquet with a `text` column.  Memory-efficient
(~1 RG at a time).  Deletes the original after successful conversion.
"""
import pyarrow as pa
import pyarrow.parquet as pq
import glob
import os
import sys

SRC_DIR = "/mnt/data/zz/datasets/github-code"
DST_DIR = "/mnt/data/zz/datasets/github-code-nanochat"

os.makedirs(DST_DIR, exist_ok=True)

sources = sorted(glob.glob(os.path.join(SRC_DIR, "train-*-of-*.parquet")))
print(f"Found {len(sources)} source shards")

for i, src in enumerate(sources, 1):
    stem = os.path.basename(src).replace("train-", "").replace("-of-01126", "")
    dst = os.path.join(DST_DIR, f"train_{stem}.parquet")

    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        sz = os.path.getsize(dst)
        print(f"  [{i}/{len(sources)}] cached {dst} ({sz/1e6:.0f} MB)")
        continue

    pf = pq.ParquetFile(src)
    n_rg = pf.num_row_groups
    n_rows = pf.metadata.num_rows
    print(f"  [{i}/{len(sources)}] {os.path.basename(src)}: {n_rows} rows, {n_rg} RGs", end="", flush=True)

    # Write one row group at a time to the output file
    writer = None
    try:
        for rg_idx in range(n_rg):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column("content").to_pylist()
            batch = pa.table({"text": pa.array(texts, type=pa.string())})

            if writer is None:
                writer = pq.ParquetWriter(dst, batch.schema)
            writer.write_table(batch)
            print(".", end="", flush=True)

        writer.close()
        sz = os.path.getsize(dst)
        print(f" done ({sz/1e6:.0f} MB)")

    except Exception as e:
        print(f" FAILED: {e}")
        if writer:
            writer.close()
        if os.path.exists(dst):
            os.remove(dst)
        sys.exit(1)

# Create a validation split from the last shard
print("\nSetting up val split...")
last_src = sources[-1]
last_stem = os.path.basename(last_src).replace("train-", "").replace("-of-01126", "")
val_dst = os.path.join(DST_DIR, f"val_{last_stem}.parquet")
if not os.path.exists(val_dst):
    # Convert last shard → val_last.parquet
    pf = pq.ParquetFile(last_src)
    n_rg = pf.num_row_groups
    writer = None
    for rg_idx in range(n_rg):
        rg = pf.read_row_group(rg_idx)
        texts = rg.column("content").to_pylist()
        batch = pa.table({"text": pa.array(texts, type=pa.string())})
        if writer is None:
            writer = pq.ParquetWriter(val_dst, batch.schema)
        writer.write_table(batch)
    writer.close()
    print(f"  Created val: {val_dst} ({os.path.getsize(val_dst)/1e6:.0f} MB)")
else:
    print(f"  Val exists: {val_dst}")

# Summary
all_dst = sorted(glob.glob(os.path.join(DST_DIR, "*.parquet")))
total_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in all_dst)
total_bytes = sum(os.path.getsize(f) for f in all_dst)
print(f"\nDone. {len(all_dst)} shards, {total_rows:,} rows, {total_bytes/1e6:.0f} MB total")