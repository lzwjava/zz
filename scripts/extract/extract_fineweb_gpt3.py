"""Extract the GPT-3 ablation FineWeb-Edu parquet shards (downloaded by
plan_and_download_fineweb_gpt3.py) into a single text file that
nanoGPT/data/fineweb/prepare.py can tokenize.

Input:  /mnt/data/zz/datasets/fineweb-edu/data__*.parquet
Output: /mnt/data/nanoGPT/data/fineweb/train_fineweb.txt
"""

import gc
import glob

import pyarrow.parquet as pq

INPUT_DIR = "/mnt/data/zz/datasets/fineweb-edu"
OUTPUT_FILE = "/mnt/data/nanoGPT/data/fineweb/train_fineweb.txt"
BATCH_SIZE = 10_000
TEXT_COLUMNS = ("text", "content", "article", "text_content", "body")
SEPARATOR = "\n\n<|endoftext|><|endoftext|><|endoftext|>\n\n"

parquet_files = sorted(glob.glob(f"{INPUT_DIR}/data__*.parquet"))

if not parquet_files:
    print(f"Error: No parquet files found in {INPUT_DIR}/data__*.parquet")
    exit(1)

print(f"Found {len(parquet_files)} parquet files (showing first 5):")
for file in parquet_files[:5]:
    print(f"  - {file}")
if len(parquet_files) > 5:
    print(f"  ... and {len(parquet_files) - 5} more")

print(f"Extracting text from all files to {OUTPUT_FILE}...")

total_rows_processed = 0
total_rows_written = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for file_idx, parquet_file in enumerate(parquet_files):
        print(
            f"\n--- Processing file {file_idx + 1}/{len(parquet_files)}: {parquet_file} ---"
        )

        try:
            pf = pq.ParquetFile(parquet_file)
            schema_names = pf.schema_arrow.names
            text_col = next((c for c in TEXT_COLUMNS if c in schema_names), None)
            if text_col is None:
                print(f"  No known text column in {schema_names}, skipping")
                continue
            if file_idx == 0:
                print(f"Columns: {schema_names}; using '{text_col}'")

            file_rows = 0
            for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=[text_col]):
                texts = batch.column(0).to_pylist()
                chunk = []
                for t in texts:
                    if t is None:
                        continue
                    s = t.strip()
                    if s:
                        chunk.append(s)
                if chunk:
                    f.write(SEPARATOR.join(chunk))
                    f.write(SEPARATOR)
                    total_rows_written += len(chunk)
                file_rows += len(texts)
                total_rows_processed += len(texts)
                if file_rows % (BATCH_SIZE * 10) == 0:
                    print(
                        f"File {file_idx + 1} - {file_rows:,} rows, "
                        f"total {total_rows_processed:,} (written {total_rows_written:,})"
                    )

            del pf
            gc.collect()
            print(f"Completed {parquet_file} ({file_rows:,} rows)")

        except Exception as e:
            print(f"Error processing {parquet_file}: {e}")
            continue

print("\n\nEXTRACTION COMPLETE")
print(f"Processed {len(parquet_files)} parquet files")
print(f"Total rows read:    {total_rows_processed:,}")
print(f"Total rows written: {total_rows_written:,}")
print(f"Output saved to: {OUTPUT_FILE}")
