#!/usr/bin/env python3
"""Download the nvidia/Nemotron-ClimbMix dataset with parallel workers.

Three download modes:
  1. Main dataset: 100 .tokenized.jsonl files (part_0..part_99), ~20 GB each, ~2 TB total
  2. Small subset: 100 .tokenized.parquet files (climbmix_small/shard_0..shard_99), ~500 MB each
  3. Full: everything (main + small + nanoGPT + detokenize script)

All files served via HuggingFace LFS.  Uses wget -c (resume) so re-runs are safe.

Usage:
    # Small subset only (fast smoke test, ~50 GB)
    python3 download.py --subset small

    # Main dataset only (full ~2 TB)
    python3 download.py --subset main

    # Everything
    python3 download.py --subset full

    # Only download specific files by index range
    python3 download.py --subset main --shard-range 0-4

    # 16 parallel workers with hf_transfer (2-5x faster per stream)
    HF_HUB_ENABLE_HF_TRANSFER=1 python3 download.py --subset small --workers 16

Output: /mnt/data/zz/datasets/climbmix/
"""

import argparse
import concurrent.futures
import os
import subprocess
import sys
import threading
import time

REPO_ID = "nvidia/Nemotron-ClimbMix"

# Main dataset: part_N.tokenized.jsonl (100 files, ~20 GB each)
MAIN_FILES = [f"part_{i}.tokenized.jsonl" for i in range(100)]

# Small subset: climbmix_small/shard_N.tokenized.parquet (100 files)
SMALL_FILES = [f"climbmix_small/shard_{i}.tokenized.parquet" for i in range(100)]

# Supplementary files
EXTRA_FILES = [
    "README.md",
    "detokenize_climbmix.py",
]

_progress_lock = threading.Lock()


def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def get_file_sizes(files):
    """Batch-lookup file sizes from HF."""
    from huggingface_hub import HfApi

    api = HfApi()
    sizes = {fn: 0 for fn in files}
    entries = list(api.list_repo_tree(REPO_ID, repo_type="dataset"))
    for entry in entries:
        # Only RepoFile entries have .size (RepoFolder does not)
        if type(entry).__name__ == "RepoFile" and entry.path in sizes:
            sizes[entry.path] = entry.size or 0
    return sizes


def download_one(args):
    """Download a single file via wget. Returns (fn, bytes_downloaded, error_str, skipped)."""
    fn, expected, output_dir = args
    dest = os.path.join(output_dir, fn)

    # Ensure parent directory exists (for subdirectories like climbmix_small/)
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if os.path.exists(dest) and os.path.getsize(dest) > expected * 0.9:
        return (fn, os.path.getsize(dest), None, True)  # skipped (cached)

    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{fn}"

    try:
        result = subprocess.run(
            ["wget", "-q", "-c", "-O", dest, url],
            timeout=3600,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return (fn, 0, result.stderr[:200], False)
        actual = os.path.getsize(dest)
        return (fn, actual, None, False)
    except subprocess.TimeoutExpired:
        return (fn, 0, "TIMEOUT", False)
    except Exception as e:
        return (fn, 0, str(e), False)


def resolve_files(subset, shard_range=None):
    """Resolve file list based on subset and optional shard range."""
    files = []
    extra_files = list(EXTRA_FILES)

    if subset == "small":
        files = list(SMALL_FILES)
    elif subset == "main":
        files = list(MAIN_FILES)
    elif subset == "full":
        files = list(MAIN_FILES) + list(SMALL_FILES)
    else:
        print(f"ERROR: Unknown subset '{subset}'")
        sys.exit(1)

    # Filter by shard range if specified
    if shard_range is not None:
        try:
            lo, hi = shard_range.split("-")
            lo, hi = int(lo), int(hi)
        except ValueError:
            print(f"ERROR: Invalid --shard-range '{shard_range}'. Use format '0-4'")
            sys.exit(1)

        def in_range(fn):
            # Extract the numerical index from the filename
            for pat in ("part_", "shard_"):
                if pat in fn:
                    try:
                        idx = int(fn.split(pat)[1].split(".")[0])
                        return lo <= idx <= hi
                    except (ValueError, IndexError):
                        return True
            return True

        files = [f for f in files if in_range(fn=f)]

    return files + extra_files


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--subset",
        choices=["small", "main", "full"],
        default="small",
        help="Which subset to download (default: small)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel downloads (default: 8)",
    )
    p.add_argument(
        "--shard-range",
        metavar="LO-HI",
        help="Only download shards in index range, e.g. '0-4'",
    )
    p.add_argument(
        "--output-dir",
        default="/home/xiaoxin/projects/zz/datasets/climbmix",
        help="Output directory",
    )
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = resolve_files(args.subset, args.shard_range)

    print(f"Looking up {len(files)} file sizes...", flush=True)
    file_sizes = get_file_sizes(files)
    total_available = sum(file_sizes.values())
    print(
        f"  {human_bytes(total_available)} total | {args.workers} workers",
        flush=True,
    )

    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER"):
        print("  hf_transfer enabled (expect 2-5x faster per-stream)")
    print()

    # Build task list
    tasks = [(fn, file_sizes.get(fn, 0), args.output_dir) for fn in files]

    # Stats tracking (thread-safe via lock)
    t0 = time.time()
    done_bytes = 0
    done_count = 0
    skip_count = 0
    err_count = 0
    total = len(tasks)

    print(f"Downloading {total} files with {args.workers} parallel workers...")
    print()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(download_one, t): t[0] for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            fn, size, err, skipped = future.result()
            with _progress_lock:
                if skipped:
                    skip_count += 1
                    done_bytes += size
                elif err:
                    err_count += 1
                    print(f"  FAIL {fn}: {err[:80]}", flush=True)
                else:
                    done_count += 1
                    done_bytes += size

                elapsed = time.time() - t0
                rate = done_bytes / elapsed / 1024 / 1024 if elapsed > 0 else 0
                remaining = total - done_count - skip_count - err_count
                print(
                    f"  [{done_count + skip_count + err_count}/{total}] "
                    f"{fn}: {human_bytes(size):>8} | "
                    f"{human_bytes(done_bytes):>8} total | "
                    f"{rate:.1f} MB/s | "
                    f"{remaining} remaining",
                    flush=True,
                )

    elapsed = time.time() - t0
    print()
    print(f"Done. {human_bytes(done_bytes)} in {args.output_dir}")
    print(
        f"  {done_count} downloaded, {skip_count} cached, "
        f"{err_count} errors, {elapsed / 60:.1f} min"
    )
    if err_count > 0:
        print("  Re-run to retry failed files (wget -c resumes partials)")
    print()

    if args.subset == "small":
        print("Dataset is already in parquet format (text column).")
        print("Next step: tokenize with tokenize.py")
    else:
        print("Main dataset uses .tokenized.jsonl format (LFS, tokenized).")
        print("Use detokenize_climbmix.py to convert back to text.")


if __name__ == "__main__":
    main()