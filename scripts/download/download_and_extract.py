#!/usr/bin/env python3
"""Download FineWeb parquet shards and extract text with pyarrow iter_batches (memory-safe)."""

import os
import urllib.request
import pyarrow.parquet as pq
import gc

BASE_URL = "https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/data/CC-MAIN-2013-20"
SHARDS = [f"000_000{i:02d}.parquet" for i in range(21)]
OUTPUT_FILE = "fineweb_extracted_all.txt"
TEMP_DIR = "fineweb_parquet"

os.makedirs(TEMP_DIR, exist_ok=True)


def download_shard(filename):
    url = f"{BASE_URL}/{filename}?download=true"
    dest = os.path.join(TEMP_DIR, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"  Already exists: {filename}")
        return dest
    print(f"  Downloading {filename}...", flush=True)
    urllib.request.urlretrieve(url, dest)
    size_mb = os.path.getsize(dest) / 1024 / 1024
    print(f"  Downloaded: {size_mb:.1f} MB", flush=True)
    return dest


total_docs = 0

with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
    for shard in SHARDS:
        print(f"\n=== Processing {shard} ===", flush=True)
        try:
            path = download_shard(shard)
            pf = pq.ParquetFile(path)
            print(f"  Row groups: {pf.metadata.num_row_groups}", flush=True)

            # Use iter_batches — streams small RecordBatches, not full row groups
            for batch in pf.iter_batches(batch_size=4096):
                text_col = batch.column("text")
                for i in range(len(text_col)):
                    text = str(text_col[i].as_py()).strip()
                    if text:
                        out.write(text + "\n<|endoftext|>\n")
                        total_docs += 1

                if total_docs % 50000 == 0:
                    print(f"  Docs: {total_docs:,}", flush=True)
                    out.flush()

            os.remove(path)
            gc.collect()
            print(f"  Shard done. Total docs: {total_docs:,}", flush=True)

        except Exception as e:
            print(f"  Error: {e}", flush=True)
            continue

print(f"\nAll done! Total docs: {total_docs:,}", flush=True)
