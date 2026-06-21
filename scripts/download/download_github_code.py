#!/usr/bin/env python3
"""Download ~100GB of code from codeparrot/github-code.

The dataset has 1126 parquet shards (~285MB each, ~320GB total).
This script downloads them sequentially via huggingface_hub, resumable.

Usage:
    # Download ~100GB (all 30 languages)
    python3.11 scripts/download/download_github_code.py

    # Custom size
    python3.11 scripts/download/download_github_code.py --target-gb 50

Output: /mnt/data/zz/datasets/github-code/train-00000-of-01126.parquet ...

After download, tokenize with:
    python3.11 scripts/extract/tokenize_github_code.py
"""

import argparse
import os
import time

from huggingface_hub import HfApi

REPO_ID = "codeparrot/github-code"


def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def list_shards():
    """Return list of (path, size) for all parquet shards, sorted."""
    api = HfApi()
    shards = []
    for f in api.list_repo_tree(REPO_ID, repo_type="dataset", path_in_repo="data"):
        if f.path.endswith(".parquet"):
            shards.append((f.path, f.size or 0))
    shards.sort()
    return shards


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-gb", type=float, default=100,
                   help="Stop after downloading this many GB (default: 100)")
    p.add_argument("--output-dir", default="/mnt/data/zz/datasets/github-code",
                   help="Output directory")
    args = p.parse_args()

    target_bytes = int(args.target_gb * 1024**3)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Listing shards on HuggingFace...", flush=True)
    shards = list_shards()
    total_available = sum(s for _, s in shards)
    print(f"  {len(shards)} shards, {human_bytes(total_available)} total")

    # Figure out how many shards to download
    selected = []
    running = 0
    for path, size in shards:
        if running >= target_bytes:
            break
        selected.append((path, size))
        running += size
    print(f"Plan: download {len(selected)} shards (~{human_bytes(running)})")
    print()

    # Download via wget (fastest, resumable, no HF SDK overhead)
    import subprocess
    t0 = time.time()
    done_bytes = 0
    skipped = 0
    errors = 0
    for i, (path, size) in enumerate(selected, 1):
        fname = os.path.basename(path)
        dest = os.path.join(args.output_dir, fname)

        # Skip if already downloaded (>90% of expected size)
        if os.path.exists(dest) and os.path.getsize(dest) > size * 0.9:
            skipped += 1
            done_bytes += os.path.getsize(dest)
            if i % 100 == 0:
                elapsed = time.time() - t0
                rate = done_bytes / elapsed / 1024 / 1024 if elapsed > 0 else 0
                print(f"  [{i}/{len(selected)}] cached | "
                      f"{human_bytes(done_bytes)} | {rate:.1f} MB/s", flush=True)
            continue

        url = (f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{path}")
        print(f"  [{i}/{len(selected)}] {fname}...", end="", flush=True)
        try:
            result = subprocess.run(
                ["wget", "-q", "-c", "-O", dest, url],
                timeout=300, capture_output=True, text=True)
            if result.returncode != 0:
                print(f" wget error: {result.stderr[:100]}")
                errors += 1
                continue
            actual = os.path.getsize(dest)
            done_bytes += actual
            elapsed = time.time() - t0
            rate = done_bytes / elapsed / 1024 / 1024 if elapsed > 0 else 0
            pct = done_bytes / target_bytes * 100
            print(f" {human_bytes(actual)} | "
                  f"{human_bytes(done_bytes)} ({pct:.0f}%) | "
                  f"{rate:.1f} MB/s", flush=True)
        except subprocess.TimeoutExpired:
            print(f" TIMEOUT")
            errors += 1
        except Exception as e:
            print(f" ERROR: {e}")
            errors += 1

        if done_bytes >= target_bytes:
            print(f"\nReached target {human_bytes(done_bytes)}.")
            break

    elapsed = time.time() - t0
    print(f"\nDone. {human_bytes(done_bytes)} in {args.output_dir}")
    print(f"  {len(selected) - skipped - errors} downloaded, "
          f"{skipped} cached, {errors} errors, {elapsed/60:.1f} min total")
    print(f"\nNext step — tokenize for nanoGPT:")
    print(f"  python3.11 scripts/extract/tokenize_github_code.py")


if __name__ == "__main__":
    main()
