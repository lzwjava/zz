#!/usr/bin/env python3
"""Download the codeparrot/codeparrot-clean-train dataset (~12.7 GB gzipped).

Python-only source code, ~50 GB uncompressed, 5.3M files across 53 shards.

Dataset: https://huggingface.co/datasets/codeparrot/codeparrot-clean-train

Files are named file-000000000001.json.gz .. file-000000000053.json.gz.
Each is NDJSON (one JSON object per line, each = one file's content + metadata).

Usage:
    # Download all 53 shards
    python3.11 download.py

    # Valid set only (smoke test, 1 shard, ~142 MB)
    python3.11 download.py --valid-only

    # Download only 5 GB smoke test
    python3.11 download.py --target-gb 5

Output: /mnt/data/zz/datasets/codeparrot-clean/file-000000000001.json.gz ...
"""

import argparse
import os
import subprocess
import time

REPO_ID = "codeparrot/codeparrot-clean"
TRAIN_FILES = [f"file-{i:012d}.json.gz" for i in range(1, 54)]  # 53 shards
VALID_FILE = "file-000000000054.json.gz"


def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def get_file_size(filename):
    """Get size of a single file from HF repo listing."""
    from huggingface_hub import HfApi
    api = HfApi()
    if filename == VALID_FILE:
        repo = "codeparrot/codeparrot-clean-valid"
    else:
        repo = REPO_ID
    for f in api.list_repo_tree(repo, repo_type="dataset"):
        if f.path == filename:
            return f.size or 0
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-gb", type=float, default=0,
                   help="Stop after downloading this many GB (default: all)")
    p.add_argument("--valid-only", action="store_true",
                   help="Download only the valid set (1 shard, smoke test)")
    p.add_argument("--output-dir", default="/mnt/data/zz/datasets/codeparrot-clean",
                   help="Output directory")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.valid_only:
        files = [VALID_FILE]
    else:
        files = TRAIN_FILES + [VALID_FILE]

    print("Looking up file sizes...", flush=True)
    file_sizes = {}
    for fn in files:
        file_sizes[fn] = get_file_size(fn)
    total_available = sum(file_sizes.values())
    print(f"  {len(files)} files, {human_bytes(total_available)} total (gzipped)")
    print()

    selected = files[:]
    if args.target_gb > 0:
        target_bytes = int(args.target_gb * 1024**3)
        selected = []
        running = 0
        for fn in files:
            if running >= target_bytes:
                break
            selected.append(fn)
            running += file_sizes.get(fn, 0)
        print(f"Plan: download {len(selected)} files (~{human_bytes(running)})")
    else:
        running = total_available

    print()

    t0 = time.time()
    done_bytes = 0
    skipped = 0
    errors = 0

    for i, fn in enumerate(selected, 1):
        expected = file_sizes.get(fn, 0)
        dest = os.path.join(args.output_dir, fn)

        if fn == VALID_FILE:
            repo = "codeparrot/codeparrot-clean-valid"
        else:
            repo = REPO_ID

        if os.path.exists(dest) and os.path.getsize(dest) > expected * 0.9:
            skipped += 1
            done_bytes += os.path.getsize(dest)
            if i % 10 == 0:
                elapsed = time.time() - t0
                rate = done_bytes / elapsed / 1024 / 1024 if elapsed > 0 else 0
                print(f"  [{i}/{len(selected)}] cached | "
                      f"{human_bytes(done_bytes)} | {rate:.1f} MB/s", flush=True)
            continue

        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{fn}"
        print(f"  [{i}/{len(selected)}] {fn}...", end="", flush=True)

        try:
            result = subprocess.run(
                ["wget", "-q", "--show-progress", "-c", "-O", dest, url],
                timeout=600, capture_output=True, text=True)
            if result.returncode != 0:
                print(f" wget error: {result.stderr[:200]}")
                errors += 1
                continue

            actual = os.path.getsize(dest)
            done_bytes += actual
            elapsed = time.time() - t0
            rate = done_bytes / elapsed / 1024 / 1024 if elapsed > 0 else 0
            pct_str = ""
            if args.target_gb > 0:
                pct = done_bytes / (args.target_gb * 1024**3) * 100
                pct_str = f" ({pct:.0f}%)"
            print(f" {human_bytes(actual)} | "
                  f"{human_bytes(done_bytes)}{pct_str} | "
                  f"{rate:.1f} MB/s", flush=True)

        except subprocess.TimeoutExpired:
            print(f" TIMEOUT")
            errors += 1
        except Exception as e:
            print(f" ERROR: {e}")
            errors += 1

        if args.target_gb > 0 and done_bytes >= args.target_gb * 1024**3:
            print(f"\nReached target {human_bytes(done_bytes)}.")
            break

    elapsed = time.time() - t0
    downloaded = len(selected) - skipped - errors
    print(f"\nDone. {human_bytes(done_bytes)} in {args.output_dir}")
    print(f"  {downloaded} downloaded, {skipped} cached, "
          f"{errors} errors, {elapsed/60:.1f} min")
    print(f"\nNext step:")
    print(f"  python3.11 convert.py")


if __name__ == "__main__":
    main()