#!/usr/bin/env python3
"""
Search the notes RAG index (retrieval only, no LLM).

Usage:
    python search.py "how to fine-tune Qwen3 with LoRA"
    python search.py "正则表达式" --k 5
    python search.py "vllm" --lang en --json
    python search.py --interactive

Defaults to hybrid retrieval: dense (BAAI/bge-m3) + BM25 fused with RRF.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))

from retriever import DEFAULT_EMBED_MODEL, Retriever  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Search the notes RAG index")
    p.add_argument("query", nargs="?", help="Query text (omit with --interactive)")
    p.add_argument("--index-dir", default=str(DIR / "rag_index"))
    p.add_argument("--model", default=None, help="Override the embedding model stored in the index")
    p.add_argument("--device", default=None)
    p.add_argument("--k", type=int, default=8)
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="RRF weight for dense results: 1.0 = dense only, 0.0 = BM25 only",
    )
    p.add_argument("--lang", default=None, help="Filter: en | zh")
    p.add_argument("--no-bm25", action="store_true")
    p.add_argument("--no-faiss", action="store_true", help="Force NumPy dense search")
    p.add_argument("--snippet", type=int, default=400, help="Chars of each chunk to print")
    p.add_argument("--json", action="store_true", help="Machine-readable output")
    p.add_argument("--interactive", "-i", action="store_true")
    return p.parse_args()


def render(hits: list[dict], snippet: int) -> None:
    for i, hit in enumerate(hits, 1):
        label = hit["title"] + (f" › {hit['heading']}" if hit["heading"] else "")
        text = hit["text"][:snippet].replace("\n", " ").strip()
        if len(hit["text"]) > snippet:
            text += " …"
        raw = " ".join(
            f"{k}={hit[k]:.3f}" for k in ("dense", "bm25") if k in hit
        )
        print(f"\n[{i}] {label}  (rrf={hit['score']:.4f}{(' ' + raw) if raw else ''})")
        print(f"    {hit['file']}  [{hit['lang']}]")
        print(f"    {text}")



def main():
    args = parse_args()
    retriever = Retriever(
        Path(args.index_dir),
        model_name=args.model,
        device=args.device,
        use_bm25=not args.no_bm25,
        use_faiss=not args.no_faiss,
    )
    print(
        f"Index: {retriever.meta['count']:,} chunks | model: {retriever.model_name} | "
        f"faiss: {retriever.faiss_index is not None} | bm25: {retriever.bm25 is not None}",
        file=sys.stderr,
    )

    def run(query: str) -> None:
        hits = retriever.hybrid(query, k=args.k, alpha=args.alpha, lang=args.lang)
        if args.json:
            print(json.dumps(hits, ensure_ascii=False, indent=2))
        else:
            print(f"\nQuery: {query}  ({len(hits)} hits)")
            render(hits, args.snippet)

    if args.interactive or not args.query:
        print("Interactive search — empty line or Ctrl-D to quit.", file=sys.stderr)
        while True:
            try:
                query = input("\nquery> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not query:
                continue
            run(query)
    else:
        run(args.query)


if __name__ == "__main__":
    main()
