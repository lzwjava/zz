"""Tokenize the GPT-3 ablation FineWeb-Edu parquet shards straight into
uint16 token shards (skipping the intermediate train_fineweb.txt).

Reads:  /mnt/data/zz/datasets/fineweb-edu/data__*.parquet
Writes: /mnt/data/nanoGPT/data/fineweb/edu_fineweb100B/edufineweb_{split}_{idx:06d}.npy

Output format matches nanoGPT/data/fineweb/prepare_fineweb.py so the
existing training loader works unchanged (shard 0 -> val, rest -> train,
100M uint16 tokens per shard, GPT-2 BPE, each doc prefixed with <|endoftext|>).
"""

import argparse
import glob
import multiprocessing as mp
import os

import numpy as np
import pyarrow.parquet as pq
import tiktoken
from tqdm import tqdm

INPUT_DIR = "/mnt/data/zz/datasets/fineweb-edu"
OUTPUT_DIR = "/mnt/data/nanoGPT/data/fineweb/edu_fineweb100B"
SHARD_SIZE = int(1e8)  # 100M tokens per shard
BATCH_SIZE = 10_000  # parquet rows per batch (main process)
POOL_CHUNKSIZE = 16  # docs per worker chunk
TEXT_COLUMNS = ("text", "content", "article", "text_content", "body")

enc = tiktoken.get_encoding("gpt2")
EOT = enc._special_tokens["<|endoftext|>"]


def tokenize(doc):
    tokens = [EOT]
    tokens.extend(enc.encode_ordinary(doc))
    arr = np.array(tokens, dtype=np.int64)
    assert (arr >= 0).all() and (arr < 2**16).all(), "token id exceeds uint16"
    return arr.astype(np.uint16)


def iter_docs(parquet_files):
    for file_idx, path in enumerate(parquet_files):
        try:
            pf = pq.ParquetFile(path)
        except Exception as e:
            print(f"[skip] {path}: {e}")
            continue
        names = pf.schema_arrow.names
        text_col = next((c for c in TEXT_COLUMNS if c in names), None)
        if text_col is None:
            print(f"[skip] {path}: no text column in {names}")
            continue
        print(
            f"[{file_idx + 1}/{len(parquet_files)}] {os.path.basename(path)} "
            f"(col={text_col})"
        )
        for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=[text_col]):
            for t in batch.column(0).to_pylist():
                if t is None:
                    continue
                s = t.strip()
                if s:
                    yield s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=INPUT_DIR)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    ap.add_argument("--nprocs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    existing = sorted(glob.glob(os.path.join(args.output_dir, "edufineweb_*.npy")))
    if existing:
        print(
            f"WARNING: {len(existing)} shard(s) already in {args.output_dir}; "
            f"this run will overwrite from shard 0."
        )

    parquet_files = sorted(glob.glob(os.path.join(args.input_dir, "data__*.parquet")))
    if not parquet_files:
        print(f"Error: no parquet files in {args.input_dir}")
        return 1
    print(
        f"Found {len(parquet_files)} parquet files; tokenizing with "
        f"{args.nprocs} workers, {args.shard_size:,} tokens/shard"
    )

    buf = np.empty((args.shard_size,), dtype=np.uint16)
    token_count = 0
    shard_index = 0
    progress = None

    def flush_shard(buf, count, idx):
        split = "val" if idx == 0 else "train"
        fname = os.path.join(args.output_dir, f"edufineweb_{split}_{idx:06d}")
        np.save(fname, buf[:count])
        return fname

    with mp.Pool(args.nprocs) as pool:
        for tokens in pool.imap(
            tokenize, iter_docs(parquet_files), chunksize=POOL_CHUNKSIZE
        ):
            if progress is None:
                progress = tqdm(
                    total=args.shard_size,
                    unit="tok",
                    unit_scale=True,
                    desc=f"Shard {shard_index}",
                )

            n = len(tokens)
            if token_count + n < args.shard_size:
                buf[token_count : token_count + n] = tokens
                token_count += n
                progress.update(n)
            else:
                remainder = args.shard_size - token_count
                buf[token_count : token_count + remainder] = tokens[:remainder]
                progress.update(remainder)
                progress.close()
                fname = flush_shard(buf, args.shard_size, shard_index)
                print(f"  wrote {fname}.npy")
                shard_index += 1
                progress = None

                leftover = tokens[remainder:]
                # safety: a single doc bigger than shard_size would corrupt buf
                assert len(leftover) <= args.shard_size, (
                    f"document with {n} tokens exceeds shard size {args.shard_size}"
                )
                buf[: len(leftover)] = leftover
                token_count = len(leftover)

    if token_count > 0:
        if progress is not None:
            progress.close()
        fname = flush_shard(buf, token_count, shard_index)
        print(f"  wrote {fname}.npy ({token_count:,} tokens, partial)")
        shard_index += 1

    print(f"\nDone. {shard_index} shard(s) in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
