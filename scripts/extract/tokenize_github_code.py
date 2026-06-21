#!/usr/bin/env python3
"""Tokenize downloaded github-code parquet files into nanoGPT binary format.

Reads all .parquet files from /mnt/data/zz/datasets/github-code/
Tokenizes 'code' column with GPT-2 BPE (tiktoken), produces:
  /mnt/data/zz/datasets/github-code-tok/train_XXXXXX.npy  (100M tokens each)
  /mnt/data/zz/datasets/github-code-tok/val_000000.npy    (first shard = val)

Usage:
    # After download_github_code.py finishes:
    python3.11 scripts/extract/tokenize_github_code.py

    # Custom paths
    python3.11 scripts/extract/tokenize_github_code.py \
        --input-dir /mnt/data/zz/datasets/github-code \
        --output-dir /mnt/data/zz/datasets/github-code-tok
"""

import argparse
import glob
import multiprocessing as mp
import os
import time

import numpy as np
import pyarrow.parquet as pq
import tiktoken

SHARD_SIZE = int(1e8)  # 100M tokens per shard

enc = tiktoken.get_encoding("gpt2")
EOT = enc._special_tokens["<|endoftext|>"]


def tokenize_file(pq_path):
    """Tokenize one parquet file. Returns np.uint16 array."""
    table = pq.read_table(pq_path, columns=["content"])
    all_tokens = []
    for code in table.column("content"):
        if code is None:
            continue
        text = code.as_py()
        if not text or len(text.strip()) < 10:
            continue
        tokens = [EOT] + enc.encode_ordinary(text)
        all_tokens.extend(tokens)
    if not all_tokens:
        return np.array([], dtype=np.uint16)
    tokens_np = np.array(all_tokens, dtype=np.int64)
    assert tokens_np.max() < 2**16, f"Token value {tokens_np.max()} exceeds uint16"
    return tokens_np.astype(np.uint16)


def write_shard(filename, tokens_np):
    np.save(filename, tokens_np)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default="/mnt/data/zz/datasets/github-code")
    p.add_argument("--output-dir", default="/mnt/data/zz/datasets/github-code-tok")
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    parquet_files = sorted(glob.glob(os.path.join(args.input_dir, "*.parquet")))
    if not parquet_files:
        print(f"ERROR: No .parquet files in {args.input_dir}")
        print("Run download_github_code.py first.")
        return

    total_size = sum(os.path.getsize(f) for f in parquet_files)
    print(f"Found {len(parquet_files)} parquet files ({total_size / 1024**3:.1f} GB)")

    nprocs = args.workers or max(1, os.cpu_count() // 2)
    print(f"Tokenizing with {nprocs} workers...", flush=True)

    t0 = time.time()
    shard_index = 0
    all_tokens_np = np.empty((SHARD_SIZE,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    files_done = 0

    with mp.Pool(nprocs) as pool:
        for tokens in pool.imap(tokenize_file, parquet_files, chunksize=1):
            files_done += 1
            if len(tokens) == 0:
                continue

            if token_count + len(tokens) < SHARD_SIZE:
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
            else:
                # Write current shard
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(args.output_dir,
                                        f"{split}_{shard_index:06d}.npy")
                remainder = SHARD_SIZE - token_count
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                write_shard(filename, all_tokens_np)
                shard_index += 1

                # Start new shard with leftover
                leftover = len(tokens) - remainder
                all_tokens_np[0:leftover] = tokens[remainder:]
                token_count = leftover

            if files_done % 20 == 0:
                elapsed = time.time() - t0
                total_tok = shard_index * SHARD_SIZE + token_count
                rate = total_tok / elapsed if elapsed > 0 else 0
                print(f"  {files_done}/{len(parquet_files)} files | "
                      f"{shard_index + 1} shards | "
                      f"{total_tok / 1e9:.2f}B tokens | "
                      f"{rate / 1e6:.1f}M tok/s", flush=True)

    # Write remaining tokens
    if token_count > 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(args.output_dir, f"{split}_{shard_index:06d}.npy")
        write_shard(filename, all_tokens_np[:token_count])
        shard_index += 1

    elapsed = time.time() - t0
    total_tok = (shard_index - 1) * SHARD_SIZE + token_count
    print(f"\nDone. {shard_index} shards, ~{total_tok / 1e9:.2f}B tokens, "
          f"{elapsed / 60:.1f} min")
    print(f"Output: {args.output_dir}/")
    print(f"\nTo train with nanoGPT, update config/train_fineweb_760m.py:")
    print(f'  shard_dir = "{args.output_dir}"')


if __name__ == "__main__":
    main()
