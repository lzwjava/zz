#!/usr/bin/env python3
"""Download the codeparrot/codeparrot-clean dataset with parallel workers.

Python-only source code, ~12.7 GB gzipped across 53 train shards + 1 valid shard.

Usage:
    # Full download (default: 8 parallel workers)
    python3.11 download.py

    # 16 parallel workers
    python3.11 download.py --workers 16

    # Valid set only (smoke test)
    python3.11 download.py --valid-only

    # With hf_transfer (2-5x faster)
    HF_HUB_ENABLE_HF_TRANSFER=1 python3.11 download.py --workers 16

Output: /mnt/data/zz/datasets/codeparrot-clean/file-*.json.gz
"""

import argparse
import concurrent.futures
import os
import subprocess
import threading
import time

REPO_ID = "codeparrot/codeparrot-clean"
TRAIN_FILES = [f"file-{i:012d}.json.gz" for i in range(1, 54)]
VALID_FILE = "file-000000000054.json.gz"

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
    sizes = {}
    for fn in files:
        repo = "codeparrot/codeparrot-clean-valid" if fn == VALID_FILE else REPO_ID
        for f in api.list_repo_tree(repo, repo_type="dataset"):
            if f.path == fn:
                sizes[fn] = f.size or 0
                break
    return sizes


def download_one(args):
    """Download a single file via wget. Returns (fn, bytes_downloaded, error_str)."""
    fn, expected, output_dir = args
    dest = os.path.join(output_dir, fn)

    if fn == VALID_FILE:
        repo = "codeparrot/codeparrot-clean-valid"
    else:
        repo = REPO_ID

    if os.path.exists(dest) and os.path.getsize(dest) > expected * 0.9:
        return (fn, os.path.getsize(dest), None, True)  # skipped

    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{fn}"

    try:
        result = subprocess.run(
            ["wget", "-q", "-c", "-O", dest, url],
            timeout=600, capture_output=True, text=True)
        if result.returncode != 0:
            return (fn, 0, result.stderr[:200], False)
        actual = os.path.getsize(dest)
        return (fn, actual, None, False)
    except subprocess.TimeoutExpired:
        return (fn, 0, "TIMEOUT", False)
    except Exception as e:
        return (fn, 0, str(e), False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel downloads (default: 8)")
    p.add_argument("--valid-only", action="store_true",
                   help="Download only the valid set")
    p.add_argument("--output-dir",
                   default="/mnt/data/zz/datasets/codeparrot-clean",
                   help="Output directory")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.valid_only:
        files = [VALID_FILE]
    else:
        files = TRAIN_FILES + [VALID_FILE]

    print(f"Looking up {len(files)} file sizes...", flush=True)
    file_sizes = get_file_sizes(files)
    total_available = sum(file_sizes.values())
    print(f"  {human_bytes(total_available)} total (gzipped), "
          f"{args.workers} workers", flush=True)

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
                    flush=True
                )

    elapsed = time.time() - t0
    print()
    print(f"Done. {human_bytes(done_bytes)} in {args.output_dir}")
    print(f"  {done_count} downloaded, {skip_count} cached, "
          f"{err_count} errors, {elapsed/60:.1f} min")
    if err_count > 0:
        print(f"  Re-run to retry failed files (wget -c resumes partials)")
    print()
    print("Next step:")
    print("  python3.11 convert.py")


if __name__ == "__main__":
    main()