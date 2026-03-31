#!/usr/bin/env python3
"""Download FineWeb dataset (streaming) and save to text file.
Run with --limit N to cap documents (useful for testing).
Without --limit, downloads the full sample-10BT subset."""

import argparse
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=None, help="Max documents to download")
parser.add_argument("--output", type=str, default="fineweb_extracted_all.txt", help="Output file")
parser.add_argument("--subset", type=str, default="sample-10BT", help="FineWeb subset name")
args = parser.parse_args()

total_docs = 0

dataset = load_dataset(
    "HuggingFaceFW/fineweb",
    name=args.subset,
    split="train",
    streaming=True
)

with open(args.output, "w", encoding="utf-8") as f:
    for doc in dataset:
        f.write(doc["text"].strip())
        f.write("\n<|endoftext|>\n")
        total_docs += 1
        if total_docs % 10000 == 0:
            print(f"Processed {total_docs} docs...")
        if args.limit and total_docs >= args.limit:
            break

print(f"Done. Total docs: {total_docs}")
