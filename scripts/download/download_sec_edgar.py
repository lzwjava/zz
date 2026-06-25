#!/usr/bin/env python3
"""Download SEC-EDGAR filings from kapilrao/SEC-EDGAR on HuggingFace.

The dataset has 10 filing-type directories (10-K, 10-Q, 144, 20-F, 3, 4, 5, 8-K, S-1, S-8),
each containing parquet shards (~600MB for 10-K, ~80MB for 10-Q, etc.).
This script downloads them sequentially via wget, resumable.

Usage:
    # Download all filing types
    python3.11 scripts/download/download_sec_edgar.py

    # Download specific filing types only
    python3.11 scripts/download/download_sec_edgar.py --types 10-K 10-Q

    # Custom size limit
    python3.11 scripts/download/download_sec_edgar.py --target-gb 50

Output: /mnt/data/zz/datasets/sec-edgar/10-K/*.parquet, 10-Q/*.parquet, ...
"""

import argparse
import os
import time

from huggingface_hub import HfApi

REPO_ID = "kapilrao/SEC-EDGAR"
ALL_TYPES = ["10-K", "10-Q", "144", "20-F", "3", "4", "5", "8-K", "S-1", "S-8"]


def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def list_shards(filing_types):
    """Return list of (path, size) for parquet shards in given filing types."""
    api = HfApi()
    shards = []
    for ftype in filing_types:
        print(f"  Listing {ftype}/...", end="", flush=True)
        count = 0
        for f in api.list_repo_tree(REPO_ID, repo_type="dataset", path_in_repo=ftype):
            if f.path.endswith(".parquet"):
                shards.append((f.path, f.size or 0))
                count += 1
        print(f" {count} shards", flush=True)
    shards.sort()
    return shards


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--types", nargs="+", default=ALL_TYPES,
                   help=f"Filing types to download (default: all {len(ALL_TYPES)})")
    p.add_argument("--target-gb", type=float, default=9999,
                   help="Stop after downloading this many GB (default: unlimited)")
    p.add_argument("--output-dir", default="/mnt/data/zz/datasets/sec-edgar",
                   help="Output directory")
    args = p.parse_args()

    target_bytes = int(args.target_gb * 1024**3)
    os.makedirs(args.output_dir, exist_ok=True)

    # Create subdirs for each filing type
    for ft in args.types:
        os.makedirs(os.path.join(args.output_dir, ft), exist_ok=True)

    print("Listing shards on HuggingFace...", flush=True)
    shards = list_shards(args.types)
    total_available = sum(s for _, s in shards)
    print(f"\n  {len(shards)} total shards, {human_bytes(total_available)} total")

    # Filter to what we can fit
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
        dest = os.path.join(args.output_dir, path)  # preserves subdir structure

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

        url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{path}"
        print(f"  [{i}/{len(selected)}] {path}...", end="", flush=True)
        try:
            result = subprocess.run(
                ["wget", "-q", "-c", "-O", dest, url],
                timeout=600, capture_output=True, text=True)
            if result.returncode != 0:
                print(f" wget error: {result.stderr[:100]}")
                errors += 1
                continue
            actual = os.path.getsize(dest)
            done_bytes += actual
            elapsed = time.time() - t0
            rate = done_bytes / elapsed / 1024 / 1024 if elapsed > 0 else 0
            pct = done_bytes / target_bytes * 100 if target_bytes < 9999 * 1024**3 else 0
            print(f" {human_bytes(actual)} | "
                  f"{human_bytes(done_bytes)} | "
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


if __name__ == "__main__":
    main()
