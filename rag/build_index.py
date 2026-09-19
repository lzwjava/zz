#!/usr/bin/env python3
"""
Build the RAG index for lzwjava's notes (en + zh).

Steps:
    1. Load posts from the Jekyll blog (or a finetune JSONL)
    2. Clean + chunk Markdown
    3. Embed chunks (sentence-transformers, GPU if available)
    4. Save dense index (FAISS flat IP + embeddings.npy)
    5. Save sparse BM25 index (bm25.py)

Usage:
    python build_index.py                          # full corpus, BAAI/bge-m3
    python build_index.py --limit-posts 200        # quick smoke test
    python build_index.py --model intfloat/multilingual-e5-small   # ~5x faster
    python build_index.py --no-bm25 --no-faiss     # dense only, numpy search
    python build_index.py --from-jsonl ../finetune/notes_sft.jsonl

Output: rag_index/ (chunks.jsonl, embeddings.npy, index.faiss, bm25_*, index_meta.json)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DIR = Path(__file__).parent
sys.path.insert(0, str(DIR))

from retriever import (  # noqa: E402
    DEFAULT_EMBED_MODEL,
    DEFAULT_LANGS,
    DEFAULT_POSTS_ROOT,
    build_chunks,
    encoder_dim,
    iter_jsonl_corpus,
    iter_posts,
    load_encoder,
    passage_prefix,
)


def parse_args():
    p = argparse.ArgumentParser(description="Build the notes RAG index")
    p.add_argument("--posts-root", default=str(DEFAULT_POSTS_ROOT))
    p.add_argument("--langs", nargs="+", default=list(DEFAULT_LANGS))
    p.add_argument("--from-jsonl", default=None, help="Read corpus from an SFT jsonl instead of the blog")
    p.add_argument("--out-dir", default=str(DIR / "rag_index"))
    p.add_argument("--model", default=DEFAULT_EMBED_MODEL, help="sentence-transformers model")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-seq-length", type=int, default=0, help="Override model max_seq_length")
    p.add_argument("--chunk-size", type=int, default=900)
    p.add_argument("--chunk-overlap", type=int, default=150)
    p.add_argument("--min-chunk", type=int, default=60)
    p.add_argument("--limit-posts", type=int, default=0, help="Only use the first N posts (debug)")
    p.add_argument("--max-chunks", type=int, default=0, help="Stop after N chunks (debug)")
    p.add_argument("--fp16", action="store_true", help="Store embeddings as float16 on disk")
    p.add_argument("--no-bm25", action="store_true", help="Skip the sparse BM25 index")
    p.add_argument("--no-faiss", action="store_true", help="Skip the FAISS index (numpy search)")
    return p.parse_args()


def encode_chunks(encoder, model_name: str, chunks: list[dict], batch_size: int, fp16: bool) -> np.ndarray:
    prefix = passage_prefix(model_name)
    texts = [prefix + c["embed_text"] for c in chunks]
    total = len(texts)
    dim = encoder_dim(encoder)
    out = np.empty((total, dim), dtype=np.float16 if fp16 else np.float32)

    t0 = time.time()
    done = 0
    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        vecs = encoder.encode(
            batch,
            batch_size=len(batch),
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        out[start : start + len(batch)] = vecs.astype(out.dtype, copy=False)
        done += len(batch)

        if (start // batch_size) % 10 == 0 or done == total:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed else 0.0
            eta = (total - done) / rate if rate else 0.0
            print(
                f"  embedded {done:,}/{total:,} chunks "
                f"({rate:,.0f}/s, ETA {eta/60:.1f} min)",
                flush=True,
            )
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:      {args.model}")
    print(f"Output:     {out_dir}")
    print(f"Chunking:   {args.chunk_size} chars / {args.chunk_overlap} overlap")
    print()

    # 1-2. Corpus -> chunks -------------------------------------------------
    print("Loading corpus...", flush=True)
    if args.from_jsonl:
        posts = iter_jsonl_corpus(Path(args.from_jsonl))
    else:
        posts = iter_posts(Path(args.posts_root), args.langs)

    if args.limit_posts:
        def _head(gen, n):
            for i, item in enumerate(gen):
                if i >= n:
                    break
                yield item

        posts = _head(posts, args.limit_posts)

    chunks = build_chunks(
        posts,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_len=args.min_chunk,
        max_chunks=args.max_chunks or None,
    )
    if not chunks:
        raise SystemExit("No chunks produced — check --posts-root / --from-jsonl")
    print(f"\nTotal chunks: {len(chunks):,}")

    by_lang: dict[str, int] = {}
    for c in chunks:
        by_lang[c["lang"]] = by_lang.get(c["lang"], 0) + 1
    print("  by lang: " + ", ".join(f"{k}={v:,}" for k, v in sorted(by_lang.items())))
    avg_len = sum(len(c["text"]) for c in chunks) / len(chunks)
    print(f"  avg chunk: {avg_len:,.0f} chars")

    # 3. Embed --------------------------------------------------------------
    print(f"\nLoading encoder ({args.model})...", flush=True)
    encoder = load_encoder(args.model, device=args.device, max_seq_length=args.max_seq_length or None)
    dim = encoder_dim(encoder)
    device = getattr(encoder, "device", "unknown")
    print(f"  dim={dim}, device={device}")

    print("Embedding chunks...", flush=True)
    embeddings = encode_chunks(encoder, args.model, chunks, args.batch_size, args.fp16)

    # 4. Save dense index ---------------------------------------------------
    np.save(out_dir / "embeddings.npy", embeddings)
    print(f"  wrote embeddings.npy {embeddings.shape} {embeddings.dtype}")

    faiss_ok = False
    if not args.no_faiss:
        try:
            import faiss

            index = faiss.IndexFlatIP(dim)
            # FAISS requires float32; cast in blocks to limit peak memory.
            block = 100_000
            for start in range(0, len(chunks), block):
                index.add(np.asarray(embeddings[start : start + block], dtype="float32"))
            faiss.write_index(index, str(out_dir / "index.faiss"))
            faiss_ok = True
            print(f"  wrote index.faiss ({index.ntotal:,} vectors)")
        except ImportError:
            print("  [warn] faiss not installed — numpy search will be used")

    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"  wrote chunks.jsonl ({len(chunks):,} records)")

    # 5. Sparse BM25 --------------------------------------------------------
    bm25_ok = False
    if not args.no_bm25:
        print("\nBuilding BM25 index...", flush=True)
        from bm25 import BM25Index, tokenize

        t0 = time.time()
        bm = BM25Index()
        bm.add_documents((tokenize(c["embed_text"]) for c in chunks))
        bm.save(out_dir)
        bm25_ok = True
        print(f"  done in {time.time() - t0:.1f}s")

    # 6. Metadata -----------------------------------------------------------
    meta = {
        "model": args.model,
        "dim": int(dim),
        "count": len(chunks),
        "by_lang": by_lang,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "min_chunk": args.min_chunk,
        "source": str(args.from_jsonl or args.posts_root),
        "langs": list(args.langs) if not args.from_jsonl else [],
        "embedding_dtype": str(embeddings.dtype),
        "faiss": faiss_ok,
        "bm25": bm25_ok,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "index_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone. Index in {out_dir}/")
    print(f"  Search:  python search.py \"your query\"")
    print(f"  Chat:    python rag_chat.py --question \"your question\"")


if __name__ == "__main__":
    main()
