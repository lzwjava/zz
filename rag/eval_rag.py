#!/usr/bin/env python3
"""
Retrieval eval: can we find the source post for held-out titles?

Uses finetune/notes_sft_eval.jsonl (200 held-out posts). For each example the
post title is the query and the post file is the gold answer; we measure how
often a chunk from the gold post appears in the top-k.

Usage:
    python eval_rag.py                                  # hybrid, k=10, 200 examples
    python eval_rag.py --mode dense                     # dense only
    python eval_rag.py --mode bm25 --k 5
    python eval_rag.py --n 50 --show-misses 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))

from retriever import Retriever  # noqa: E402

DEFAULT_EVAL = DIR.parent / "finetune" / "notes_sft_eval.jsonl"


def parse_args():
    p = argparse.ArgumentParser(description="Measure RAG retrieval quality")
    p.add_argument("--eval-jsonl", default=str(DEFAULT_EVAL))
    p.add_argument("--index-dir", default=str(DIR / "rag_index"))
    p.add_argument("--model", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n", type=int, default=0, help="Limit examples (0 = all)")
    p.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--show-misses", type=int, default=5)
    return p.parse_args()


def load_examples(path: Path, n: int):
    examples = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            convs = rec.get("conversations") or []
            meta = rec.get("meta") or {}
            if len(convs) < 2 or not meta.get("file"):
                continue
            examples.append({"query": convs[0]["content"], "gold": meta["file"]})
    return examples[:n] if n else examples


def main():
    args = parse_args()
    examples = load_examples(Path(args.eval_jsonl), args.n)
    if not examples:
        raise SystemExit(f"No usable examples in {args.eval_jsonl}")

    retriever = Retriever(Path(args.index_dir), model_name=args.model, device=args.device)
    print(f"Index: {retriever.meta['count']:,} chunks | model: {retriever.model_name}")
    print(f"Eval:  {len(examples)} queries | mode: {args.mode} | k={args.k}\n")

    hits_at_1 = hits_at_k = 0
    mrr = 0.0
    misses = []

    for i, ex in enumerate(examples):
        if args.mode == "dense":
            pairs = retriever.dense(ex["query"], args.k)
            ranked = [retriever.chunks[idx]["file"] for idx, _ in pairs]
        elif args.mode == "bm25":
            pairs = retriever.sparse(ex["query"], args.k)
            ranked = [retriever.chunks[idx]["file"] for idx, _ in pairs]
        else:
            ranked = [h["file"] for h in retriever.hybrid(ex["query"], k=args.k, alpha=args.alpha)]

        if ex["gold"] in ranked:
            hits_at_k += 1
            rank = ranked.index(ex["gold"]) + 1
            mrr += 1.0 / rank
            if rank == 1:
                hits_at_1 += 1
        else:
            misses.append((ex["query"], ex["gold"], ranked[:3]))

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(examples)} ... recall@{args.k}={hits_at_k/(i+1):.3f}")

    n = len(examples)
    print("\n" + "=" * 60)
    print(f"Recall@{args.k}: {hits_at_k / n:.3f}  ({hits_at_k}/{n})")
    print(f"Recall@1:    {hits_at_1 / n:.3f}  ({hits_at_1}/{n})")
    print(f"MRR:         {mrr / n:.3f}")
    print(f"Misses:      {len(misses)}")

    if misses and args.show_misses:
        print(f"\nFirst {min(args.show_misses, len(misses))} misses:")
        for query, gold, top3 in misses[: args.show_misses]:
            print(f"\n  Q: {query[:80]}")
            print(f"  gold: {gold}")
            print(f"  top3: {', '.join(top3)}")


if __name__ == "__main__":
    main()
