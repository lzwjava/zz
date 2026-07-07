#!/usr/bin/env python3
"""Download FineWeb-Edu for GPT-3 ablation on AMD MI300X (US / direct HuggingFace).

Adapted from plan_and_download_fineweb_gpt3.py for non-China environments:
  - Uses huggingface.co directly (no mirror)
  - No proxy warnings
  - Smaller default (--tokens) to leave room on disk
  - Resumable via progress.json (same logic)

Run from the repo root (cd ~/zz), not from this script's directory —
the default --output-dir is a relative path (datasets/fineweb-edu).

Usage:
  # Default: ~25B tokens (~100 GB)
  cd ~/zz
  python3 scripts/download/h200/download_fineweb.py

  # Custom amount
  python3 scripts/download/h200/download_fineweb.py --tokens 50000000000

  # Force re-list shards from the Hub
  python3 scripts/download/h200/download_fineweb.py --refresh-plan
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from typing import List

from huggingface_hub import HfApi

REPO_ID = "HuggingFaceFW/fineweb-edu"
DEFAULT_TOKENS = 25_000_000_000  # 25B tokens (~100 GB) — for AMD MI300X
BYTES_PER_TOKEN = 4.0
PROGRESS_FILE = "progress.json"


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


def load_progress(path: str) -> dict:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_progress(path: str, state: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def build_or_load_state(output_dir: str, refresh: bool, target_tokens: int) -> dict:
    progress_path = os.path.join(output_dir, PROGRESS_FILE)
    state = None if refresh else load_progress(progress_path)
    if state and state.get("repo_id") == REPO_ID:
        print(
            f"Loaded plan from {progress_path} ({len(state['shards'])} shards).",
            flush=True,
        )
        return state

    print(f"Listing shards in {REPO_ID}...", flush=True)
    all_shards = list_parquet_shards(REPO_ID)
    if not all_shards:
        print("No shards found.", file=sys.stderr)
        sys.exit(1)

    target_bytes = int(target_tokens * BYTES_PER_TOKEN)
    selected = select_shards(all_shards, target_bytes)
    state = {
        "repo_id": REPO_ID,
        "target_tokens": target_tokens,
        "bytes_per_token": BYTES_PER_TOKEN,
        "shards": [
            {
                "path": s["path"],
                "size": s["size"],
                "dump": s["dump"],
                "status": "pending",
            }
            for s in selected
        ],
    }
    os.makedirs(output_dir, exist_ok=True)
    save_progress(progress_path, state)
    print(f"Wrote new plan to {progress_path}.", flush=True)
    return state


def download_one(url: str, dest: str, retries: int = 3) -> int:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    if os.path.exists(dest):
        return os.path.getsize(dest)
    existing = os.path.getsize(tmp) if os.path.exists(tmp) else 0

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


def print_plan_summary(state: dict) -> None:
    shards = state["shards"]
    total = sum(s["size"] for s in shards)
    done = [s for s in shards if s["status"] == "done"]
    done_bytes = sum(s["size"] for s in done)
    est_tokens = int(total / BYTES_PER_TOKEN)
    dumps = sorted({s["dump"] for s in shards})
    target = state.get("target_tokens", DEFAULT_TOKENS)
    print(
        f"\nPlan: ~{target:,} tokens "
        f"@ {BYTES_PER_TOKEN} bytes/token (GPT-3 ablation):",
        flush=True,
    )
    print(f"  shards:        {len(shards)}  ({len(done)} done)", flush=True)
    print(
        f"  download size: {human_bytes(total)}  "
        f"({human_bytes(done_bytes)} already on disk)",
        flush=True,
    )
    print(f"  est. tokens:   {est_tokens:,}", flush=True)
    print(
        f"  dumps covered: {len(dumps)}"
        + (f" ({dumps[0]} .. {dumps[-1]})" if len(dumps) > 1 else f" ({dumps[0]})"),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    p.add_argument("--output-dir", default="datasets/fineweb-edu")
    p.add_argument(
        "--tokens",
        type=int,
        default=DEFAULT_TOKENS,
        help=f"Target tokens to download (default: {DEFAULT_TOKENS:,} = ~{int(DEFAULT_TOKENS * BYTES_PER_TOKEN / 1e9)} GB)",
    )
    p.add_argument(
        "--refresh-plan",
        action="store_true",
        help="Re-list shards from the Hub and overwrite progress.json plan",
    )
    args = p.parse_args()

    state = build_or_load_state(args.output_dir, args.refresh_plan, args.tokens)
    progress_path = os.path.join(args.output_dir, PROGRESS_FILE)
    print_plan_summary(state)

    print(f"\nDownloading to {args.output_dir} from huggingface.co ...", flush=True)
    shards = state["shards"]
    total_bytes = sum(s["size"] for s in shards)
    done_bytes = sum(s["size"] for s in shards if s["status"] == "done")
    t0 = time.time()
    bytes_this_run = 0

    for i, sh in enumerate(shards, 1):
        dest = os.path.join(args.output_dir, sh["path"].replace("/", "__"))
        if sh["status"] == "done" and os.path.exists(dest):
            print(
                f"  [{i}/{len(shards)}] done  {sh['path']} ({human_bytes(sh['size'])})",
                flush=True,
            )
            continue

        url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{sh['path']}?download=true"
        print(
            f"  [{i}/{len(shards)}] {sh['path']} ({human_bytes(sh['size'])})",
            flush=True,
        )
        size = download_one(url, dest)

        sh["status"] = "done"
        save_progress(progress_path, state)
        done_bytes += size
        bytes_this_run += size
        elapsed = max(time.time() - t0, 1e-3)
        rate = bytes_this_run / elapsed / 1024 / 1024
        remaining = total_bytes - done_bytes
        eta = remaining / max(bytes_this_run / elapsed, 1) if bytes_this_run else 0
        print(
            f"      progress: {human_bytes(done_bytes)} / "
            f"{human_bytes(total_bytes)}  "
            f"{rate:.1f} MB/s  ETA {eta / 60:.1f} min",
            flush=True,
        )

    print(
        f"\nDone. {len(shards)} shards, {human_bytes(done_bytes)} on disk, "
        f"~{int(done_bytes / BYTES_PER_TOKEN):,} tokens.",
        flush=True,
    )


if __name__ == "__main__":
    main()