#!/usr/bin/env python3
"""Convert sec-edgar parquets (multi-column with `text` col) → nanochat format.

The source parquets have many columns but already contain a `text` column.
We extract just that column into single-column parquets for nanochat.
"""
import pyarrow as pa
import pyarrow.parquet as pq
import glob
import os

SRC_DIR = "/mnt/data/zz/datasets/sec-edgar"
DST_DIR = "/mnt/data/zz/datasets/sec-edgar-nanochat"

os.makedirs(DST_DIR, exist_ok=True)

all_sources = sorted(glob.glob(os.path.join(SRC_DIR, "**/*.parquet"), recursive=True))
print(f"Found {len(all_sources)} source files")

written = 0
for i, src in enumerate(all_sources, 1):
    rel = os.path.relpath(src, SRC_DIR)  # e.g. "10-K/uuid.parquet"
    stem = rel.replace("/", "_").replace(".parquet", "")
    dst = os.path.join(DST_DIR, f"{stem}.parquet")

    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        sz = os.path.getsize(dst)
        print(f"  [{i}/{len(all_sources)}] cached {stem} ({sz/1e6:.0f} MB)")
        continue

    pf = pq.ParquetFile(src)
    n_rg = pf.num_row_groups
    n_rows = pf.metadata.num_rows
    print(f"  [{i}/{len(all_sources)}] {rel}: {n_rows} rows, {n_rg} RGs", end="", flush=True)

    writer = None
    try:
        # Handle potentially corrupt files
        for rg_idx in range(n_rg):
            try:
                rg = pf.read_row_group(rg_idx)
            except Exception as e:
                print(f" SKIP (corrupt RG {rg_idx}: {e})", end="", flush=True)
                continue
            texts = rg.column("text").to_pylist()
            batch = pa.table({"text": pa.array(texts, type=pa.string())})
            if writer is None:
                writer = pq.ParquetWriter(dst, batch.schema)
            writer.write_table(batch)
            print(".", end="", flush=True)
        writer.close()
        sz = os.path.getsize(dst)
        print(f" done ({sz/1e6:.0f} MB)")
        written += 1
    except Exception as e:
        print(f" FAILED: {e}")
        if writer:
            writer.close()
        if os.path.exists(dst):
            os.remove(dst)
        raise

print(f"\nDone. {written} files written to {DST_DIR}")