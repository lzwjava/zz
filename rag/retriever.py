#!/usr/bin/env python3
"""
Shared retrieval utilities for the lzwjava notes RAG pipeline.

Corpus:  ~/projects/jekyll-ai-blog/_posts/{en,zh}/*.md
         (or any JSONL produced by finetune/build_dataset.py)
Index:   rag_index/  -> chunks.jsonl + embeddings.npy [+ index.faiss] [+ bm25_*]

The markdown cleaning rules mirror finetune/build_dataset.py so the RAG
corpus and the SFT corpus stay aligned.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable, Iterator, Sequence

DIR = Path(__file__).parent
if str(DIR) not in sys.path:  # allow `from bm25 import ...` from any cwd
    sys.path.insert(0, str(DIR))

# ---------------------------------------------------------------------------
# Markdown cleaning (same rules as finetune/build_dataset.py)
# ---------------------------------------------------------------------------

LIQUID = re.compile(r"\{%.*?%\}|\{\{.*?\}\}", re.DOTALL)
KRAMDOWN = re.compile(r"\{:\s*\.[^}]*\}")
IMG_REF = re.compile(r"!\[.*?\]\(.*?\)")
CAPTION = re.compile(r"^\*Source:.*$", re.MULTILINE)
BLANKS = re.compile(r"\n{3,}")

DEFAULT_POSTS_ROOT = Path.home() / "projects/jekyll-ai-blog/_posts"
DEFAULT_LANGS = ("en", "zh")

DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
FAST_EMBED_MODEL = "intfloat/multilingual-e5-small"


def clean_body(text: str) -> str:
    """Strip Jekyll/Liquid tags, kramdown attrs and image refs from a post."""
    text = LIQUID.sub("", text)
    text = KRAMDOWN.sub("", text)
    text = IMG_REF.sub("", text)
    text = CAPTION.sub("", text)
    text = BLANKS.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Front matter
# ---------------------------------------------------------------------------

def _parse_simple_yaml(raw: str) -> dict:
    """Tiny YAML subset parser used when PyYAML is not installed."""
    meta: dict = {}
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line or line.startswith((" ", "\t", "-")):
            continue
        key, _, value = line.partition(":")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif value.lower() in ("true", "false"):
            value = value.lower() == "true"
        meta[key.strip()] = value
    return meta


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Return (metadata, body) for a Jekyll post."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    raw, body = text[3:end], text[end + 4 :].lstrip("\n")
    try:  # prefer a real parser when available
        import yaml

        meta = yaml.safe_load(raw) or {}
        if not isinstance(meta, dict):
            meta = _parse_simple_yaml(raw)
    except Exception:
        meta = _parse_simple_yaml(raw)
    return meta, body


def post_slug(path: Path, meta: dict) -> str:
    """Best-effort Jekyll slug for a post file (YYYY-MM-DD-slug.md)."""
    if meta.get("slug"):
        return str(meta["slug"])
    stem = path.stem
    return re.sub(r"^\d{4}-\d{2}-\d{2}-", "", stem)


def post_url(path: Path, meta: dict) -> str:
    """Best-effort URL. `permalink: /:title` is the blog default for posts."""
    if meta.get("permalink"):
        return str(meta["permalink"])
    return f"/{post_slug(path, meta)}/"


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def iter_posts(posts_root: Path = DEFAULT_POSTS_ROOT, langs: Sequence[str] = DEFAULT_LANGS) -> Iterator[dict]:
    """Yield one dict per markdown post: file/lang/title/meta/body."""
    for lang in langs:
        lang_dir = Path(posts_root) / lang
        if not lang_dir.exists():
            print(f"  [{lang}] directory not found, skipping")
            continue
        count = 0
        for f in sorted(lang_dir.rglob("*.md")):
            meta, body = parse_frontmatter(f.read_text(encoding="utf-8", errors="replace"))
            title = str(meta.get("title", "")).strip()
            if not title:
                continue
            body = clean_body(body)
            if not body:
                continue
            count += 1
            yield {
                "file": str(f.relative_to(Path(posts_root).parent)),
                "lang": lang,
                "title": title,
                "url": post_url(f, meta),
                "type": str(meta.get("type", "post")),
                "generated": bool(meta.get("generated", False)),
                "body": body,
            }
        print(f"  [{lang}] loaded {count} posts")


def iter_jsonl_corpus(path: Path) -> Iterator[dict]:
    """Yield posts from a `{"conversations": [...]}` SFT jsonl file."""
    count = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            convs = rec.get("conversations") or []
            if len(convs) < 2:
                continue
            title = convs[0]["content"].strip()
            body = clean_body(convs[1]["content"])
            if not title or not body:
                continue
            meta = rec.get("meta", {}) or {}
            count += 1
            yield {
                "file": meta.get("file", f"jsonl:{count}"),
                "lang": meta.get("lang", "unknown"),
                "title": title,
                "url": "",
                "type": str(meta.get("type", "post")),
                "generated": bool(meta.get("generated", False)),
                "body": body,
            }
    print(f"  [jsonl] loaded {count} records")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading_path, body) sections."""
    matches = list(HEADING.finditer(text))
    if not matches:
        return [("", text)]

    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0 and text[: matches[0].start()].strip():
        sections.append(("", text[: matches[0].start()].strip()))

    stack: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        level, title = len(m.group(1)), m.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end() : end].strip()
        sections.append((" > ".join(t for _, t in stack), body))
    return sections


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    """Hard-split an oversized paragraph near natural boundaries."""
    out: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window_start = start + size // 2
            for sep in ("\n", "。", ". ", "; ", " "):
                cut = text.rfind(sep, window_start, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return out


def chunk_text(
    text: str,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    min_len: int = 60,
) -> list[tuple[str, str]]:
    """Pack markdown into (heading_path, chunk) pairs of ~chunk_size chars."""
    chunks: list[tuple[str, str]] = []

    for heading, body in split_sections(text):
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", body) if p.strip()]
        buf = ""

        def flush() -> None:
            nonlocal buf
            if buf and len(buf) >= min_len:
                chunks.append((heading, buf))
            elif buf:
                # short tail: keep it only if it has substance
                if chunks and chunks[-1][0] == heading and len(buf) + len(chunks[-1][1]) < chunk_size * 2:
                    chunks[-1] = (heading, chunks[-1][1] + "\n\n" + buf)
            buf = ""

        for para in paragraphs:
            if len(para) > chunk_size:
                flush()
                for piece in _split_long(para, chunk_size, chunk_overlap):
                    chunks.append((heading, piece))
                continue
            if buf and len(buf) + len(para) + 2 > chunk_size:
                tail = buf[-chunk_overlap:] if chunk_overlap else ""
                tail = tail[tail.find("\n") + 1 :] if "\n" in tail else tail
                flush()
                buf = (tail + "\n\n" + para).strip() if tail else para
            else:
                buf = (buf + "\n\n" + para).strip() if buf else para
        flush()

    return [(h, c) for h, c in chunks if len(c) >= min_len]


def build_chunks(
    posts: Iterable[dict],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    min_len: int = 60,
    max_chunks: int | None = None,
) -> list[dict]:
    """Turn posts into embedding-ready chunk records."""
    chunks: list[dict] = []
    for post in posts:
        for i, (heading, text) in enumerate(
            chunk_text(post["body"], chunk_size, chunk_overlap, min_len)
        ):
            # Context prefix helps both dense retrieval and the reader.
            prefix = post["title"] + (f" — {heading}" if heading else "")
            chunks.append(
                {
                    "id": len(chunks),
                    "file": post["file"],
                    "lang": post["lang"],
                    "title": post["title"],
                    "url": post["url"],
                    "type": post["type"],
                    "generated": post["generated"],
                    "heading": heading,
                    "chunk_index": i,
                    "embed_text": f"{prefix}\n\n{text}",
                    "text": text,
                }
            )
            if max_chunks and len(chunks) >= max_chunks:
                return chunks
    return chunks


# ---------------------------------------------------------------------------
# Embedding model helpers
# ---------------------------------------------------------------------------

def is_e5(model_name: str) -> bool:
    return "e5" in model_name.lower()


def passage_prefix(model_name: str) -> str:
    return "passage: " if is_e5(model_name) else ""


def query_prefix(model_name: str) -> str:
    return "query: " if is_e5(model_name) else ""


def load_encoder(model_name: str, device: str | None = None, max_seq_length: int | None = None):
    """Load a SentenceTransformer; imports lazily so --help stays fast."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    if max_seq_length:
        model.max_seq_length = max_seq_length
    return model


def encoder_dim(encoder) -> int:
    """Embedding size across sentence-transformers versions."""
    for attr in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        if hasattr(encoder, attr):
            return int(getattr(encoder, attr)())
    raise AttributeError("encoder exposes no embedding-dimension method")


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """Dense (FAISS / NumPy) + optional sparse (BM25) retrieval with RRF fusion."""

    def __init__(
        self,
        index_dir: Path,
        model_name: str | None = None,
        device: str | None = None,
        use_bm25: bool = True,
        use_faiss: bool = True,
    ):
        self.index_dir = Path(index_dir)
        meta_path = self.index_dir / "index_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No index at {self.index_dir}. Run: python build_index.py"
            )
        self.meta = json.loads(meta_path.read_text())
        self.model_name = model_name or self.meta["model"]

        self.chunks = [
            json.loads(line)
            for line in (self.index_dir / "chunks.jsonl").open(encoding="utf-8")
            if line.strip()
        ]

        self.encoder = None
        self._device = device

        self.faiss_index = None
        self.embeddings = None
        faiss_path = self.index_dir / "index.faiss"
        if use_faiss and faiss_path.exists():
            try:
                import faiss

                self.faiss_index = faiss.read_index(str(faiss_path))
            except Exception as exc:  # pragma: no cover - optional dependency
                print(f"[warn] could not load FAISS index: {exc}")
        if self.faiss_index is None:
            import numpy as np

            self.embeddings = np.load(self.index_dir / "embeddings.npy", mmap_mode="r")

        self.bm25 = None
        if use_bm25 and (self.index_dir / "bm25_meta.json").exists():
            from bm25 import BM25Index

            self.bm25 = BM25Index.load(self.index_dir)

        self._dense_dim = (
            self.faiss_index.d if self.faiss_index is not None else self.embeddings.shape[1]
        )

    # -- encoding -----------------------------------------------------------
    def encode_query(self, query: str):
        if self.encoder is None:
            self.encoder = load_encoder(self.model_name, device=self._device)
        import numpy as np

        vec = self.encoder.encode(
            [query_prefix(self.model_name) + query],
            normalize_embeddings=True,
            batch_size=1,
            show_progress_bar=False,
        )
        return np.asarray(vec, dtype="float32")

    # -- retrieval ----------------------------------------------------------
    def dense(self, query: str, k: int, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        q = self.encode_query(query)
        if self.faiss_index is not None:
            scores, ids = self.faiss_index.search(q, k if allowed is None else min(len(self.chunks), k * 5))
            pairs = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
        else:
            import numpy as np

            # Chunk the matmul to keep peak memory low on large indexes.
            n = self.embeddings.shape[0]
            top_scores = None
            top_ids = None
            step = 200_000
            for start in range(0, n, step):
                block = np.asarray(self.embeddings[start : start + step], dtype="float32")
                block_scores = block @ q[0]
                if top_scores is None:
                    top_scores, top_ids = block_scores, np.arange(start, start + block.shape[0])
                    keep = min(k, top_scores.size)
                    sel = np.argpartition(-top_scores, keep - 1)[:keep]
                    top_scores, top_ids = top_scores[sel], top_ids[sel]
                else:
                    cand_scores = np.concatenate([top_scores, block_scores])
                    cand_ids = np.concatenate([top_ids, np.arange(start, start + block.shape[0])])
                    keep = min(k, cand_scores.size)
                    sel = np.argpartition(-cand_scores, keep - 1)[:keep]
                    top_scores, top_ids = cand_scores[sel], cand_ids[sel]
            order = np.argsort(-top_scores)
            pairs = [(int(top_ids[i]), float(top_scores[i])) for i in order]
        if allowed is not None:
            pairs = [(i, s) for i, s in pairs if i in allowed]
        return pairs[:k]

    def sparse(self, query: str, k: int, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        if self.bm25 is None:
            return []
        pairs = self.bm25.search(query, k if allowed is None else k * 5)
        if allowed is not None:
            pairs = [(i, s) for i, s in pairs if i in allowed]
        return pairs[:k]

    def hybrid(
        self,
        query: str,
        k: int = 8,
        alpha: float = 0.5,
        lang: str | None = None,
        pool: int = 200,
    ) -> list[dict]:
        """alpha=1.0 -> dense only, alpha=0.0 -> BM25 only, else RRF fusion."""
        allowed = None
        if lang:
            allowed = {c["id"] for c in self.chunks if c["lang"] == lang}

        lists: list[tuple[float, list[tuple[int, float]], str]] = []
        if alpha > 0:
            lists.append((alpha, self.dense(query, pool, allowed), "dense"))
        if alpha < 1 and self.bm25 is not None:
            lists.append((1.0 - alpha, self.sparse(query, pool, allowed), "bm25"))

        fused: dict[int, float] = {}
        raw: dict[int, dict] = {}
        for weight, pairs, name in lists:
            for rank, (idx, score) in enumerate(pairs):
                fused[idx] = fused.get(idx, 0.0) + weight * (1.0 / (60 + rank))
                raw.setdefault(idx, {})[name] = score

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        hits = []
        for idx, rrf in ranked:
            chunk = dict(self.chunks[idx])
            chunk["score"] = rrf
            chunk.update(raw.get(idx, {}))
            hits.append(chunk)
        return hits

    # -- rendering ----------------------------------------------------------
    def format_context(self, hits: Sequence[dict], max_chars: int = 1200) -> str:
        parts = []
        for i, hit in enumerate(hits, 1):
            text = hit["text"]
            if len(text) > max_chars:
                text = text[:max_chars].rstrip() + " …"
            label = hit["title"] + (f" › {hit['heading']}" if hit["heading"] else "")
            parts.append(f"[{i}] {label} ({hit['file']})\n{text}")
        return "\n\n---\n\n".join(parts)
