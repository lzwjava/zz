#!/usr/bin/env python3
"""SPGISpeech dataset explorer — usage: python3 explore.py [S|M|L|dev|test]"""

import sys, io, soundfile as sf
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

fs = HfFileSystem()
cfg = sys.argv[1] if len(sys.argv) > 1 else "S"

if cfg in ("S", "M", "L"):
    splits = {
        "train": [f"train-{i:05d}-of-{6 if cfg=='S' else 10 if cfg=='M' else 46:05d}.parquet" for i in range(6 if cfg=='S' else 10 if cfg=='M' else 46)],
        "validation": [f"validation-{i:05d}-of-{3:05d}.parquet" for i in range(3)],
        "test": [f"test-{i:05d}-of-{3:05d}.parquet" for i in range(3)],
    }
elif cfg in ("dev", "test"):
    splits = {cfg: [f"{cfg}-{i:05d}-of-{3:05d}.parquet" for i in range(3)]}
else:
    raise ValueError(f"Unknown config: {cfg}")

for split_name, shards in splits.items():
    total_gb = 0
    total_rows = 0
    for shard in shards:
        path = f"datasets/kensho/spgispeech/{cfg}/{shard}"
        pf = pq.ParquetFile(path, filesystem=fs)
        n = pf.metadata.num_rows
        total_rows += n
        # estimate file size from listing (avoid listing cost)
        total_gb += n * 0.00035  # ~350KB per row (12s audio @ 16kHz)
    print(f"{cfg}/{split_name}: {len(shards)} shards, ~{total_rows:,} rows")

# First row group sample
if cfg in ("S", "M", "L"):
    path = f"datasets/kensho/spgispeech/{cfg}/train-00000-of-{6 if cfg=='S' else 10 if cfg=='M' else 46:05d}.parquet"
    pf = pq.ParquetFile(path, filesystem=fs)
    t = pf.read_row_group(0, columns=["wav_filename", "transcript"])
    print(f"\nFirst train shard row group 0: {len(t)} rows")
    for i in range(3):
        print(f"  [{i}] {t.column('wav_filename')[i].as_py()}")
        print(f"       {t.column('transcript')[i].as_py()[:200]}")
    # audio
    t2 = pf.read_row_group(0, columns=["audio"])
    ae = t2.column("audio")[0].as_py()
    d, sr = sf.read(io.BytesIO(ae["bytes"]))
    print(f"\nAudio sample: {len(ae['bytes'])} bytes -> shape={d.shape}, sr={sr}, dur={d.shape[0]/sr:.2f}s")
