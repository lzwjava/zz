#!/usr/bin/env python3
"""Download FineMath parquet shards from HuggingFace.

FineMath — 54B tokens of educational math text, filtered similarly to FineWeb-Edu.

Configs (total ~139 GB):
  finemath-3plus     128 shards   60.57 GB   (K-12 problem-solving math)
  finemath-4plus      64 shards   17.10 GB   (college-level problem-solving math)
  infiwebmath-3plus   64 shards   43.59 GB   (web math filtered at grade 3+ level)
  infiwebmath-4plus   32 shards   17.91 GB   (web math filtered at grade 4+ level)

Usage:
  # Download all configs
  python finemath/download_finemath.py

  # Download a specific config only
  python finemath/download_finemath.py --config finemath-3plus

  # Download via HF mirror (faster from Asia)
  python finemath/download_finemath.py --mirror hf-mirror

  # Dry-run: list shards without downloading
  python finemath/download_finemath.py --plan
"""

import argparse
import os
import subprocess
import sys
import time

from huggingface_hub import HfApi

REPO_ID = "HuggingFaceTB/finemath"

CONFIGS = [
    "finemath-3plus",
    "finemath-4plus",
    "infiwebmath-3plus",
    "infiwebmath-4plus",
]

MIRRORS = {
    "huggingface": "https://huggingface.co",
    "hf-mirror": "https://hf-mirror.com",
}


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def list_shards(config: str) -> list[dict]:
    """Return sorted list of {path, size} for all parquet shards in a config."""
    api = HfApi()
    info = api.repo_info(repo_id=REPO_ID, repo_type="dataset", files_metadata=True)
    shards = []
    for sib in info.siblings:
        path = sib.rfilename
        if not path.endswith(".parquet"):
            continue
        if not path.startswith(f"{config}/"):
            continue
        shards.append({"path": path, "size": sib.size or 0})
    shards.sort(key=lambda x: x["path"])
    return shards


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    p.add_argument(
        "--config",
        choices=CONFIGS,
        default=None,
        help="Which config to download (default: all)",
    )
    p.add_argument("--output-dir", default="datasets/finemath", help="Output directory")
    p.add_argument(
        "--mirror",
        choices=list(MIRRORS),
        default="huggingface",
        help="HuggingFace mirror (hf-mirror is faster from Asia)",
    )
    p.add_argument(
        "--plan",
        action="store_true",
        help="Print download plan and exit without downloading",
    )
    args = p.parse_args()

    configs = [args.config] if args.config else CONFIGS
    base_url = MIRRORS[args.mirror]

    # Gather all shards across selected configs
    all_shards = []
    for cfg in configs:
        print(f"Listing shards: {cfg} ...", flush=True)
        shards = list_shards(cfg)
        if not shards:
            print(f"  No shards found for {cfg}, skipping.", file=sys.stderr)
            continue
        all_shards.extend(shards)
        cfg_bytes = sum(s["size"] for s in shards)
        print(f"  {len(shards)} shards, {human_bytes(cfg_bytes)}", flush=True)

    if not all_shards:
        print("No shards found. Exiting.", file=sys.stderr)
        sys.exit(1)

    total_bytes = sum(s["size"] for s in all_shards)
    print(f"\nTotal: {len(all_shards)} shards, {human_bytes(total_bytes)}", flush=True)

    selected = all_shards
    selected_bytes = sum(s["size"] for s in selected)
    print(
        f"Plan: {len(selected)} shards, {human_bytes(selected_bytes)} to download",
        flush=True,
    )

    if args.plan:
        print("\n--plan set; exiting without downloading.", flush=True)
        return

    # Download
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nDownloading to {args.output_dir}/ via {base_url} ...", flush=True)

    t0 = time.time()
    done_bytes = 0
    skipped = 0
    errors = 0

    for i, sh in enumerate(selected, 1):
        url = f"{base_url}/datasets/{REPO_ID}/resolve/main/{sh['path']}?download=true"
        dest = os.path.join(args.output_dir, sh["path"].replace("/", "__"))

        # Skip if already downloaded (>90% of expected size)
        if os.path.exists(dest) and os.path.getsize(dest) > sh["size"] * 0.9:
            skipped += 1
            done_bytes += os.path.getsize(dest)
            if i % 20 == 0:
                elapsed = time.time() - t0
                rate = done_bytes / elapsed / 1024 / 1024 if elapsed > 0 else 0
                print(
                    f"  [{i}/{len(selected)}] cached {human_bytes(done_bytes)} "
                    f"| {rate:.1f} MB/s",
                    flush=True,
                )
            continue

        print(f"  [{i}/{len(selected)}] {sh['path']} ({human_bytes(sh['size'])})", flush=True)

        try:
            result = subprocess.run(
                ["wget", "-q", "-c", "-O", dest, url],
                timeout=600,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"    wget error: {result.stderr[:200]}", flush=True)
                errors += 1
                continue
            actual = os.path.getsize(dest)
            done_bytes += actual
            elapsed = time.time() - t0
            rate = done_bytes / elapsed / 1024 / 1024 if elapsed > 0 else 0
            eta_sec = (
                (selected_bytes - done_bytes) / max(done_bytes / elapsed, 1)
                if done_bytes
                else 0
            )
            pct = done_bytes / selected_bytes * 100
            print(
                f"    {human_bytes(actual)} — {human_bytes(done_bytes)} "
                f"({pct:.1f}%) | {rate:.1f} MB/s | "
                f"ETA {eta_sec / 60:.1f} min",
                flush=True,
            )
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT (600s)", flush=True)
            errors += 1
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            errors += 1

    elapsed = time.time() - t0
    print(f"\n{'='*60}", flush=True)
    print(f"Done! {human_bytes(done_bytes)} in {args.output_dir}/", flush=True)
    print(
        f"Downloaded: {len(selected) - skipped - errors}, "
        f"cached: {skipped}, errors: {errors}",
        flush=True,
    )
    print(f"Duration: {elapsed / 60:.1f} min", flush=True)

    # Print extraction hint
    print(f"\nNext step — extract text from parquet files:", flush=True)
    print(f"  python scripts/download/download_and_extract.py \\", flush=True)
    print(f"      --input '{args.output_dir}/finemath*gz.parquet' \\", flush=True)
    print(f"      --output finemath_extracted.txt", flush=True)

    # Summary table
    print(f"\nFineMath config sizes on disk:", flush=True)
    print(f"  {'Config':25s} {'Shards':>6s} {'Size':>10s}", flush=True)
    print(f"  {'-'*25} {'-'*6} {'-'*10}", flush=True)
    for cfg in CONFIGS:
        cfg_shards = [s for s in selected if s["path"].startswith(f"{cfg}/")]
        cfg_bytes = sum(s["size"] for s in cfg_shards)
        print(f"  {cfg:25s} {len(cfg_shards):>6d} {human_bytes(cfg_bytes):>10s}", flush=True)
    print(f"  {'-'*25} {'-'*6} {'-'*10}", flush=True)
    print(f"  {'TOTAL':25s} {len(selected):>6d} {human_bytes(done_bytes):>10s}", flush=True)


if __name__ == "__main__":
    main()