#!/usr/bin/env python3
"""View samples from SEC-EDGAR parquet files.

Usage:
    # List all downloaded shards with row counts
    python3.11 scripts/download/view_sec_edgar.py --list

    # Random sample from all shards
    python3.11 scripts/download/view_sec_edgar.py --sample

    # N random samples
    python3.11 scripts/download/view_sec_edgar.py --sample -n 3

    # View first N rows of a specific shard
    python3.11 scripts/download/view_sec_edgar.py --file 10-K/002931a9-...parquet --head 5

    # Search for a keyword in text column
    python3.11 scripts/download/view_sec_edgar.py --search "risk factors"

    # Show only metadata (no text content)
    python3.11 scripts/download/view_sec_edgar.py --sample --meta-only

    # Show text column only, with char limit
    python3.11 scripts/download/view_sec_edgar.py --sample --text-only --chars 500
"""

import argparse
import glob
import json
import os
import random
import textwrap

import pandas as pd

DATA_DIR = "/mnt/data/zz/datasets/sec-edgar"


def find_parquets(filing_type=None):
    """Find all parquet files, optionally filtered by filing type."""
    if filing_type:
        pattern = os.path.join(DATA_DIR, filing_type, "*.parquet")
    else:
        pattern = os.path.join(DATA_DIR, "*", "*.parquet")
    files = sorted(glob.glob(pattern))
    return files


def print_separator(char="=", width=80):
    print(char * width)


def print_filing(row, idx=None, text_only=False, meta_only=False, chars=2000):
    """Pretty-print a single filing."""
    print_separator()
    label = f"Row {idx}" if idx is not None else "Filing"
    meta_filer = row.get("metadata_filer", "")
    company = ""
    if meta_filer:
        try:
            filer = json.loads(meta_filer) if isinstance(meta_filer, str) else meta_filer
            company = filer.get("company-data", {}).get("conformed-name", "")
        except (json.JSONDecodeError, AttributeError):
            pass

    acc = row.get("metadata_accession-number", "N/A")
    fdate = row.get("metadata_filing-date", "N/A")
    period = row.get("metadata_period", "N/A")

    print(f"  [{label}] {company}  |  Accession: {acc}  |  Filed: {fdate}  |  Period: {period}")
    print_separator("-")

    if not text_only:
        content = str(row.get("content", ""))
        print(f"\n--- HTML content ({len(content)} chars, showing first {chars}) ---")
        print(content[:chars])
        print("...")

    if not meta_only:
        text = str(row.get("text", ""))
        limit = chars if text_only else chars // 2
        print(f"\n--- Text ({len(text)} chars, showing first {limit}) ---")
        print(text[:limit])
        print("...")

    print()


def cmd_list(files):
    """List all shards with row counts and sizes."""
    total_rows = 0
    total_size = 0
    print(f"\n{'Shard':<70} {'Rows':>7} {'Size':>10}")
    print_separator()
    for f in files:
        rel = os.path.relpath(f, DATA_DIR)
        size = os.path.getsize(f)
        try:
            df = pd.read_parquet(f, columns=["metadata_accession-number"])
            rows = len(df)
        except Exception as e:
            rows = -1
            print(f"  ERROR reading {rel}: {e}")
        size_mb = size / 1024 / 1024
        print(f"  {rel:<68} {rows:>7} {size_mb:>8.1f}MB")
        total_rows += max(rows, 0)
        total_size += size
    print_separator()
    print(f"  Total: {len(files)} shards, {total_rows:,} rows, {total_size/1024/1024/1024:.2f} GB")


def cmd_sample(files, n, text_only, meta_only, chars):
    """Show n random samples across all shards."""
    shown = 0
    attempts = 0
    max_attempts = n * 5
    while shown < n and attempts < max_attempts:
        attempts += 1
        f = random.choice(files)
        rel = os.path.relpath(f, DATA_DIR)
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            print(f"  WARN: skipping corrupt shard {rel}: {e}")
            continue
        idx = random.randint(0, len(df) - 1)
        print(f"\n  From shard: {rel}  (row {idx}/{len(df)})")
        print_filing(df.iloc[idx], idx=idx, text_only=text_only, meta_only=meta_only, chars=chars)
        shown += 1


def cmd_head(file_path, n, text_only, meta_only, chars):
    """Show first n rows of a specific file."""
    rel = os.path.relpath(file_path, DATA_DIR)
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"  ERROR: corrupt shard {rel}: {e}")
        return
    print(f"\n  Shard: {rel}  ({len(df)} rows total, showing first {n})")
    for i in range(min(n, len(df))):
        print_filing(df.iloc[i], idx=i, text_only=text_only, meta_only=meta_only, chars=chars)


def cmd_search(files, keyword, text_only, meta_only, chars):
    """Search for keyword in text column across all shards."""
    keyword_lower = keyword.lower()
    found = 0
    print(f"\n  Searching for '{keyword}' across {len(files)} shards...")
    for f in files:
        rel = os.path.relpath(f, DATA_DIR)
        try:
            df = pd.read_parquet(f, columns=["text", "metadata_accession-number",
                                              "metadata_filing-date", "metadata_filer",
                                              "metadata_period", "content"])
        except Exception as e:
            print(f"  WARN: skipping corrupt shard {rel}: {e}")
            continue
        matches = df[df["text"].str.lower().str.contains(keyword_lower, na=False)]
        if len(matches) > 0:
            print(f"\n  Shard: {rel}  — {len(matches)} matches")
            for i, (idx, row) in enumerate(matches.head(3).iterrows()):
                print_filing(row, idx=idx, text_only=text_only, meta_only=meta_only, chars=chars)
            found += len(matches)
    print_separator()
    print(f"  Total: {found} filings containing '{keyword}'")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="List all shards with row counts")
    p.add_argument("--sample", action="store_true", help="Show random samples")
    p.add_argument("--file", type=str, help="Specific shard to view (relative path, e.g. 10-K/foo.parquet)")
    p.add_argument("--head", type=int, default=5, help="Number of rows to show with --file (default: 5)")
    p.add_argument("-n", type=int, default=2, help="Number of random samples (default: 2)")
    p.add_argument("--search", type=str, help="Search keyword in text column")
    p.add_argument("--type", type=str, help="Filter by filing type (e.g. 10-K, 10-Q)")
    p.add_argument("--text-only", action="store_true", help="Show only text column (no HTML)")
    p.add_argument("--meta-only", action="store_true", help="Show only metadata (no text/HTML)")
    p.add_argument("--chars", type=int, default=2000, help="Max chars to display per field (default: 2000)")
    args = p.parse_args()

    files = find_parquets(args.type)
    if not files:
        print(f"No parquet files found in {DATA_DIR}")
        return

    if args.list:
        cmd_list(files)
    elif args.search:
        cmd_search(files, args.search, args.text_only, args.meta_only, args.chars)
    elif args.file:
        fpath = args.file if os.path.isabs(args.file) else os.path.join(DATA_DIR, args.file)
        if not os.path.exists(fpath):
            # try glob match
            matches = glob.glob(os.path.join(DATA_DIR, f"*{args.file}*"))
            if matches:
                fpath = matches[0]
            else:
                print(f"File not found: {fpath}")
                return
        cmd_head(fpath, args.head, args.text_only, args.meta_only, args.chars)
    elif args.sample:
        cmd_sample(files, args.n, args.text_only, args.meta_only, args.chars)
    else:
        # Default: show list + 1 sample
        cmd_list(files)
        print()
        cmd_sample(files, 1, False, False, 2000)


if __name__ == "__main__":
    main()
