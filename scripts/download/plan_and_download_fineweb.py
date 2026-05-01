#!/usr/bin/env python3
"""Plan and download enough FineWeb / FineWeb-Edu shards to hit a token budget.

The existing scripts in this repo only pull from a single CommonCrawl dump
(CC-MAIN-2013-20) and have no notion of "how much is enough". This one:

  1. Enumerates parquet shards across every CC snapshot in the dataset via the
     HuggingFace Hub API.
  2. Estimates tokens-per-shard using a bytes/token heuristic (override with
     --bytes-per-token if you've measured it on your own tokenizer).
  3. With --plan, prints a download plan only — no bytes pulled.
  4. Without --plan, downloads shards in order, resumes partial files, and
     stops once the token budget is hit.

Token budget cheatsheet (English BPE, ~4 bytes/token, FineWeb shards ~2.1 GB):

  10B tokens  -> ~20 shards   ~40 GB    (GPT-2-class experiment)
  100B tokens -> ~200 shards  ~400 GB   (small-scale GPT-3 ablation)
  300B tokens -> ~600 shards  ~1.2 TB   (GPT-3 paper's training token count)
  1T tokens   -> ~2000 shards ~4 TB     (Chinchilla-optimal for ~50B params)
  1.3T tokens -> full FineWeb-Edu       (~5.4 TB)

Usage:
  # Plan a 10B-token FineWeb-Edu pull
  python plan_and_download_fineweb.py --target-tokens 10B \\
      --dataset fineweb-edu --plan

  # Download from the HF mirror (faster from Asia) into datasets/
  python plan_and_download_fineweb.py --target-tokens 10B \\
      --dataset fineweb-edu --output-dir datasets/fineweb-edu --mirror hf-mirror

  # Restrict to a single CC snapshot
  python plan_and_download_fineweb.py --target-tokens 5B \\
      --dataset fineweb --dump CC-MAIN-2024-10
"""

import argparse
import os
import sys
import time
import urllib.request
from typing import List

from huggingface_hub import HfApi

DATASETS = {
    "fineweb": "HuggingFaceFW/fineweb",
    "fineweb-edu": "HuggingFaceFW/fineweb-edu",
}

MIRRORS = {
    "huggingface": "https://huggingface.co",
    "hf-mirror": "https://hf-mirror.com",
}


def parse_token_count(s: str) -> int:
    s = s.strip().upper().replace(",", "")
    mult = 1
    if s.endswith("K"):
        mult, s = 1_000, s[:-1]
    elif s.endswith("M"):
        mult, s = 1_000_000, s[:-1]
    elif s.endswith("B"):
        mult, s = 1_000_000_000, s[:-1]
    elif s.endswith("T"):
        mult, s = 1_000_000_000_000, s[:-1]
    return int(float(s) * mult)


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if n < 1024 or unit == "PB":
            return f"{n:.1f} {unit}"
        n /= 1024


def list_parquet_shards(repo_id: str, dump_filter: str = None) -> List[dict]:
    """Return list of {path, size, dump} for every parquet shard, sorted by dump+path."""
    api = HfApi()
    info = api.repo_info(repo_id=repo_id, repo_type="dataset", files_metadata=True)
    shards = []
    for sib in info.siblings:
        path = sib.rfilename
        if not path.endswith(".parquet"):
            continue
        if not path.startswith("data/"):
            continue
        parts = path.split("/")
        if len(parts) < 3:
            continue
        dump = parts[1]
        if dump_filter and dump != dump_filter:
            continue
        shards.append({"path": path, "size": sib.size or 0, "dump": dump})
    shards.sort(key=lambda x: (x["dump"], x["path"]))
    return shards


def select_shards(shards: List[dict], target_tokens: int, bytes_per_token: float) -> List[dict]:
    target_bytes = int(target_tokens * bytes_per_token)
    selected, total = [], 0
    for sh in shards:
        if total >= target_bytes:
            break
        selected.append(sh)
        total += sh["size"]
    return selected


def download_one(url: str, dest: str, retries: int = 3) -> int:
    """Resumable download via Range header. Returns final file size."""
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
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--dataset", choices=list(DATASETS), default="fineweb-edu")
    p.add_argument("--target-tokens", default="10B",
                   help="Token budget, e.g. 10B, 300B, 1T (default: 10B)")
    p.add_argument("--bytes-per-token", type=float, default=4.0,
                   help="Bytes per token (default: 4.0 for GPT-2 BPE on English)")
    p.add_argument("--dump", default=None,
                   help="Restrict to a single CC dump, e.g. CC-MAIN-2024-10")
    p.add_argument("--output-dir", default="datasets/fineweb")
    p.add_argument("--mirror", choices=list(MIRRORS), default="huggingface")
    p.add_argument("--plan", action="store_true",
                   help="Print plan and exit without downloading")
    args = p.parse_args()

    repo_id = DATASETS[args.dataset]
    target_tokens = parse_token_count(args.target_tokens)
    base_url = MIRRORS[args.mirror]

    print(f"Listing shards in {repo_id}"
          + (f" (dump={args.dump})" if args.dump else "") + "...", flush=True)
    shards = list_parquet_shards(repo_id, args.dump)
    if not shards:
        print("No shards found. Check --dump value.", file=sys.stderr)
        sys.exit(1)

    total_bytes = sum(s["size"] for s in shards)
    print(f"  available: {len(shards)} shards, {human_bytes(total_bytes)}, "
          f"~{int(total_bytes / args.bytes_per_token):,} tokens", flush=True)

    selected = select_shards(shards, target_tokens, args.bytes_per_token)
    selected_bytes = sum(s["size"] for s in selected)
    est_tokens = int(selected_bytes / args.bytes_per_token)
    print(f"\nPlan to hit ~{target_tokens:,} tokens "
          f"@ {args.bytes_per_token} bytes/token:", flush=True)
    print(f"  shards:        {len(selected)}", flush=True)
    print(f"  download size: {human_bytes(selected_bytes)}", flush=True)
    print(f"  est. tokens:   {est_tokens:,}", flush=True)
    dumps = sorted({s["dump"] for s in selected})
    print(f"  dumps covered: {len(dumps)}"
          + (f" ({dumps[0]} .. {dumps[-1]})" if len(dumps) > 1 else f" ({dumps[0]})"),
          flush=True)

    if args.plan:
        print("\n--plan set; exiting without downloading.")
        return

    print(f"\nDownloading to {args.output_dir} via {base_url} ...", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)
    done_bytes = 0
    t0 = time.time()
    for i, sh in enumerate(selected, 1):
        url = f"{base_url}/datasets/{repo_id}/resolve/main/{sh['path']}?download=true"
        dest = os.path.join(args.output_dir, sh["path"].replace("/", "__"))
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            size = os.path.getsize(dest)
            print(f"  [{i}/{len(selected)}] cached {sh['path']} ({human_bytes(size)})",
                  flush=True)
        else:
            print(f"  [{i}/{len(selected)}] {sh['path']} "
                  f"({human_bytes(sh['size'])})", flush=True)
            size = download_one(url, dest)
        done_bytes += size
        elapsed = max(time.time() - t0, 1e-3)
        rate = done_bytes / elapsed / 1024 / 1024
        eta = (selected_bytes - done_bytes) / max(done_bytes / elapsed, 1) if done_bytes else 0
        print(f"      progress: {human_bytes(done_bytes)} / "
              f"{human_bytes(selected_bytes)}  "
              f"{rate:.1f} MB/s  ETA {eta/60:.1f} min", flush=True)

    print(f"\nDone. {len(selected)} shards, {human_bytes(done_bytes)} on disk, "
          f"~{int(done_bytes / args.bytes_per_token):,} tokens.", flush=True)
    print("Next: pipe these parquet files through scripts/extract/extract_fineweb.py "
          "(point its glob at this output dir).", flush=True)


if __name__ == "__main__":
    main()
