#!/usr/bin/env python3
"""
Okapi BM25 sparse index with a CJK-aware tokenizer.

Postings are stored as flat int32 numpy arrays (memory-mappable), so the
index for ~250k chunks loads in milliseconds and stays off the heap.

Files written by save():
    bm25_vocab.txt      one term per line, term id = line number
    bm25_postings.npy   int32 [n_pairs, 2] -> (doc_id, term_freq)
    bm25_offsets.npy    int64 [n_terms + 1] -> slice in postings
    bm25_doclen.npy     int32 [n_docs]
    bm25_meta.json      n_docs, avgdl, k1, b
"""

from __future__ import annotations

import json
import math
import re
from array import array
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# ASCII words / code identifiers, plus CJK runs split into bigrams.
WORD_RE = re.compile(r"[A-Za-z0-9_]+")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+")


def tokenize(text: str) -> list[str]:
    """Lowercase ASCII tokens + CJK character bigrams."""
    text = text.lower()
    tokens = WORD_RE.findall(text)
    for run in CJK_RE.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.postings: dict[str, array] = {}
        self.doc_len = array("i")
        self.n_docs = 0
        self.avgdl = 0.0
        # loaded state
        self.vocab: dict[str, int] | None = None
        self._postings_arr: np.ndarray | None = None
        self._offsets: np.ndarray | None = None
        self._doclen_arr: np.ndarray | None = None

    # -- build --------------------------------------------------------------
    def add_documents(self, docs: Iterable[Sequence[str]]) -> None:
        for tokens in docs:
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            for term, freq in tf.items():
                self.postings.setdefault(term, array("i")).extend((self.n_docs, freq))
            self.doc_len.append(len(tokens))
            self.n_docs += 1
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0

    # -- persistence --------------------------------------------------------
    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        terms = sorted(self.postings)
        vocab = {term: i for i, term in enumerate(terms)}

        offsets = np.zeros(len(terms) + 1, dtype=np.int64)
        total = 0
        for i, term in enumerate(terms):
            offsets[i] = total
            total += len(self.postings[term]) // 2
        offsets[len(terms)] = total

        postings = np.empty((total, 2), dtype=np.int32)
        pos = 0
        for term in terms:
            raw = self.postings[term]
            pairs = np.frombuffer(raw, dtype=np.int32).reshape(-1, 2)
            postings[pos : pos + len(pairs)] = pairs
            pos += len(pairs)

        (out_dir / "bm25_vocab.txt").write_text("\n".join(terms), encoding="utf-8")
        np.save(out_dir / "bm25_postings.npy", postings)
        np.save(out_dir / "bm25_offsets.npy", offsets)
        np.save(out_dir / "bm25_doclen.npy", np.frombuffer(self.doc_len, dtype=np.int32).copy())
        (out_dir / "bm25_meta.json").write_text(
            json.dumps(
                {
                    "n_docs": self.n_docs,
                    "avgdl": self.avgdl,
                    "k1": self.k1,
                    "b": self.b,
                    "n_terms": len(terms),
                    "n_postings": int(total),
                },
                indent=2,
            )
        )
        print(
            f"  bm25: {len(terms):,} terms, {total:,} postings, "
            f"{self.n_docs:,} docs, avgdl {self.avgdl:.1f}"
        )

    @classmethod
    def load(cls, index_dir: Path) -> "BM25Index":
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / "bm25_meta.json").read_text())
        obj = cls(k1=meta["k1"], b=meta["b"])
        obj.n_docs = meta["n_docs"]
        obj.avgdl = meta["avgdl"]
        obj.vocab = {}
        with (index_dir / "bm25_vocab.txt").open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                obj.vocab[line.rstrip("\n")] = i
        obj._postings_arr = np.load(index_dir / "bm25_postings.npy", mmap_mode="r")
        obj._offsets = np.load(index_dir / "bm25_offsets.npy")
        obj._doclen_arr = np.load(index_dir / "bm25_doclen.npy", mmap_mode="r")
        return obj

    # -- search -------------------------------------------------------------
    def search(self, query: str | Sequence[str], k: int = 50) -> list[tuple[int, float]]:
        if self._postings_arr is None:
            raise RuntimeError("BM25Index is not loaded; use BM25Index.load()")
        tokens = tokenize(query) if isinstance(query, str) else list(query)
        if not tokens:
            return []

        scores = np.zeros(self.n_docs, dtype=np.float32)
        k1, b = self.k1, self.b
        avgdl = self.avgdl or 1.0

        for term in set(tokens):
            tid = self.vocab.get(term)
            if tid is None:
                continue
            lo, hi = int(self._offsets[tid]), int(self._offsets[tid + 1])
            if hi <= lo:
                continue
            post = self._postings_arr[lo:hi]
            docs = np.asarray(post[:, 0], dtype=np.int64)
            tf = np.asarray(post[:, 1], dtype=np.float32)
            df = docs.size
            idf = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
            dl = np.asarray(self._doclen_arr[docs], dtype=np.float32)
            scores[docs] += idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avgdl))

        k = min(k, scores.size)
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[scores[top] > 0]
        order = np.argsort(-scores[top])
        return [(int(top[i]), float(scores[top[i]])) for i in order]
