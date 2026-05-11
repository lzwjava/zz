#!/usr/bin/env python3
"""Extract text from FineWeb parquet files using pyarrow streaming (memory-safe)."""

import os
import pyarrow.parquet as pq
import gc

TEMP_DIR = "fineweb_parquet"
OUTPUT_FILE = "fineweb_extracted_all.txt"

total_docs = 0

with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
    for filename in sorted(os.listdir(TEMP_DIR)):
        if not filename.endswith(".parquet"):
            continue
        path = os.path.join(TEMP_DIR, filename)
        print(
            f"\n=== Processing {filename} ({os.path.getsize(path) / 1024 / 1024:.0f} MB) ==="
        )

        try:
            # Use pyarrow streaming reader — reads row groups one at a time
            pf = pq.ParquetFile(path)
            print(f"  Row groups: {pf.metadata.num_row_groups}")

            for i in range(pf.metadata.num_row_groups):
                table = pf.read_row_group(i)
                df = table.to_pandas()

                for _, row in df.iterrows():
                    text = str(row.get("text", "")).strip()
                    if text:
                        out.write(text + "\n<|endoftext|>\n")
                        total_docs += 1

                del df, table
                gc.collect()

                if (i + 1) % 5 == 0:
                    print(f"  Row group {i + 1}, docs so far: {total_docs}")

            # Delete parquet to save disk
            os.remove(path)
            print(f"  Done. Total docs: {total_docs}")

        except Exception as e:
            print(f"  Error: {e}")
            continue

print(f"\nAll done! Total docs: {total_docs}")
