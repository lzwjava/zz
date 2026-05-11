#!/usr/bin/env python3
"""Download ~100B tokens of FineWeb-Edu for a small-scale GPT-3 ablation.

Hardcoded plan: ~200 shards, ~400 GB on disk, ~100B tokens
(English BPE, ~4 bytes/token, FineWeb shards ~2.1 GB).

Usage:
  # Plan only, no bytes pulled
  python plan_and_download_fineweb_gpt3.py --plan

  # Download from the HF mirror (faster from Asia) into datasets/fineweb-edu
  python plan_and_download_fineweb_gpt3.py --mirror hf-mirror
"""

import argparse
import os
import sys
import time
import urllib.request
from typing import List

from huggingface_hub import HfApi

REPO_ID = "HuggingFaceFW/fineweb-edu"
TARGET_TOKENS = 100_000_000_000
BYTES_PER_TOKEN = 4.0

MIRRORS = {
    "huggingface": "https://huggingface.co",
    "hf-mirror": "https://hf-mirror.com",
}


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if n < 1024 or unit == "PB":
            return f"{n:.1f} {unit}"
        n /= 1024


def list_parquet_shards(repo_id: str) -> List[dict]:
    api = HfApi()
    info = api.repo_info(repo_id=repo_id, repo_type="dataset", files_metadata=True)
    shards = []
    for sib in info.siblings:
        path = sib.rfilename
        if not path.endswith(".parquet") or not path.startswith("data/"):
            continue
        parts = path.split("/")
        if len(parts) < 3:
            continue
        shards.append({"path": path, "size": sib.size or 0, "dump": parts[1]})
    shards.sort(key=lambda x: (x["dump"], x["path"]))
    return shards


def select_shards(shards: List[dict], target_bytes: int) -> List[dict]:
    selected, total = [], 0
    for sh in shards:
        if total >= target_bytes:
            break
        selected.append(sh)
        total += sh["size"]
    return selected


def download_one(url: str, dest: str, retries: int = 3) -> int:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    existing = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if os.path.exists(dest):
        return os.path.getsize(dest)

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url)
            if existing:
                req.add_header("Range", f"bytes={existing}-")
            with urllib.request.urlopen(req, timeout=60) as resp:
                mode = "ab" if existing and resp.status == 206 else "wb"
                if mode == "wb":
                    existing = 0
                with open(tmp, mode) as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            os.rename(tmp, dest)
            return os.path.getsize(dest)
        except Exception as e:
            print(f"    attempt {attempt}/{retries} failed: {e}", flush=True)
            existing = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            time.sleep(2 * attempt)
    raise RuntimeError(f"giving up on {url}")


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    p.add_argument("--output-dir", default="datasets/fineweb-edu")
    p.add_argument("--mirror", choices=list(MIRRORS), default="huggingface")
    p.add_argument(
        "--plan", action="store_true", help="Print plan and exit without downloading"
    )
    args = p.parse_args()

    base_url = MIRRORS[args.mirror]
    target_bytes = int(TARGET_TOKENS * BYTES_PER_TOKEN)

    print(f"Listing shards in {REPO_ID}...", flush=True)
    shards = list_parquet_shards(REPO_ID)
    if not shards:
        print("No shards found.", file=sys.stderr)
        sys.exit(1)

    total_bytes = sum(s["size"] for s in shards)
    print(
        f"  available: {len(shards)} shards, {human_bytes(total_bytes)}, "
        f"~{int(total_bytes / BYTES_PER_TOKEN):,} tokens",
        flush=True,
    )

    selected = select_shards(shards, target_bytes)
    selected_bytes = sum(s["size"] for s in selected)
    est_tokens = int(selected_bytes / BYTES_PER_TOKEN)
    print(
        f"\nPlan to hit ~{TARGET_TOKENS:,} tokens "
        f"@ {BYTES_PER_TOKEN} bytes/token (small-scale GPT-3 ablation):",
        flush=True,
    )
    print(f"  shards:        {len(selected)}", flush=True)
    print(f"  download size: {human_bytes(selected_bytes)}", flush=True)
    print(f"  est. tokens:   {est_tokens:,}", flush=True)
    dumps = sorted({s["dump"] for s in selected})
    print(
        f"  dumps covered: {len(dumps)}"
        + (f" ({dumps[0]} .. {dumps[-1]})" if len(dumps) > 1 else f" ({dumps[0]})"),
        flush=True,
    )

    if args.plan:
        print("\n--plan set; exiting without downloading.")
        return

    print(f"\nDownloading to {args.output_dir} via {base_url} ...", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)
    done_bytes = 0
    t0 = time.time()
    for i, sh in enumerate(selected, 1):
        url = f"{base_url}/datasets/{REPO_ID}/resolve/main/{sh['path']}?download=true"
        dest = os.path.join(args.output_dir, sh["path"].replace("/", "__"))
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            size = os.path.getsize(dest)
            print(
                f"  [{i}/{len(selected)}] cached {sh['path']} ({human_bytes(size)})",
                flush=True,
            )
        else:
            print(
                f"  [{i}/{len(selected)}] {sh['path']} ({human_bytes(sh['size'])})",
                flush=True,
            )
            size = download_one(url, dest)
        done_bytes += size
        elapsed = max(time.time() - t0, 1e-3)
        rate = done_bytes / elapsed / 1024 / 1024
        eta = (
            (selected_bytes - done_bytes) / max(done_bytes / elapsed, 1)
            if done_bytes
            else 0
        )
        print(
            f"      progress: {human_bytes(done_bytes)} / "
            f"{human_bytes(selected_bytes)}  "
            f"{rate:.1f} MB/s  ETA {eta / 60:.1f} min",
            flush=True,
        )

    print(
        f"\nDone. {len(selected)} shards, {human_bytes(done_bytes)} on disk, "
        f"~{int(done_bytes / BYTES_PER_TOKEN):,} tokens.",
        flush=True,
    )


if __name__ == "__main__":
    main()
