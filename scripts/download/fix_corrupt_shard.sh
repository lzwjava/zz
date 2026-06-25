#!/bin/bash
# Re-download corrupt SEC-EDGAR shard
DEST=/mnt/data/zz/datasets/sec-edgar/10-K/002137a1-6454-4dca-9582-93b9d177efde-90.parquet
rm -f "$DEST"
wget -O "$DEST" \
  "https://huggingface.co/datasets/kapilrao/SEC-EDGAR/resolve/main/10-K/002137a1-6454-4dca-9582-93b9d177efde-90.parquet"
