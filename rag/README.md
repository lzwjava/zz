# lzwjava notes RAG

Retrieval-augmented generation over lzwjava's ~29k Jekyll notes (en + zh) —
the same corpus used by [`../finetune`](../finetune), but retrieved instead of
baked into weights.

```text
_posts/{en,zh}/*.md ──► clean ──► chunk ──► embed (bge-m3) ──► rag_index/
                                              └── BM25 (sparse)
question ──► hybrid retrieval (dense + BM25, RRF) ──► grounded prompt ──► answer [1][2]
```

## Quick start

```bash
cd rag

# 1. Install (venv reuses the system torch/transformers)
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt

# 2. Build the index (first run downloads the embedding model ~2.3 GB)
.venv/bin/python build_index.py --limit-posts 200    # smoke test, ~1 min
.venv/bin/python build_index.py                      # full corpus

# 3. Retrieve (no LLM)
.venv/bin/python search.py "how to export a Qwen3 LoRA to GGUF"
.venv/bin/python search.py "正则表达式" --k 5 --lang zh

# 4. RAG chat (llama.cpp + the fine-tuned GGUF from ../finetune)
.venv/bin/python rag_chat.py --question "How do I merge a LoRA adapter?"
.venv/bin/python rag_chat.py --interactive

# 5. Eval retrieval on the held-out finetune titles
.venv/bin/python eval_rag.py --k 10
```

## Files

| File | Purpose |
|---|---|
| `retriever.py` | Shared code: markdown cleaning, chunking, embeddings, dense + hybrid retrieval |
| `bm25.py` | Okapi BM25 index with a CJK-aware tokenizer (mmapped numpy postings) |
| `build_index.py` | Blog/JSONL → chunks → embeddings → `rag_index/` |
| `search.py` | Retrieval CLI (dense / BM25 / hybrid, JSON output, interactive) |
| `rag_chat.py` | Retrieve + generate with citations (llama.cpp / HF / vLLM / none) |
| `eval_rag.py` | Recall@k and MRR on `finetune/notes_sft_eval.jsonl` |
| `rag_index/` | Generated index (gitignored) |

## Build options

```bash
# Faster embedding model (~5x, English+Chinese still fine)
python build_index.py --model intfloat/multilingual-e5-small

# Dense only, numpy search (no faiss / no bm25)
python build_index.py --no-faiss --no-bm25

# Reuse the SFT jsonl instead of reading the blog
python build_index.py --from-jsonl ../finetune/notes_sft.jsonl

# Tune chunking
python build_index.py --chunk-size 700 --chunk-overlap 100

# Smaller disk footprint
python build_index.py --fp16
```

| Option | Default | Notes |
|---|---|---|
| `--model` | `BAAI/bge-m3` | Multilingual, 1024-dim, 8k context — best en+zh quality |
| `--chunk-size` / `--chunk-overlap` | 900 / 150 | Chars, split on headings then paragraphs |
| `--langs` | `en zh` | `hant`/`ja` dirs are currently empty |
| `--batch-size` | 32 | Lower it if VRAM is tight |
| `--limit-posts`, `--max-chunks` | 0 | Debug knobs |

## Retrieval options

`--alpha` controls RRF fusion: `1.0` = dense only, `0.0` = BM25 only,
`0.5` = balanced (default). BM25 catches exact identifiers (`SFTConfig`,
`n_gpu_layers`), dense catches paraphrases and cross-lingual matches.

```bash
python search.py "vllm 部署" --alpha 0.0     # keyword-heavy query
python search.py "how do embeddings work" --alpha 1.0
python search.py "LoRA" --lang en --json | jq '.[0]'
```

## Chat options

```bash
# Retrieval only — inspect what the model would see
python rag_chat.py -q "正则表达式" --backend none --show-context

# HF backend instead of llama.cpp
python rag_chat.py -q "..." --backend hf --model ../finetune/lzw-notes-merged

# vLLM on a big GPU
python rag_chat.py -q "..." --backend vllm --model ../finetune/lzw-notes-merged

# More context, longer answers
python rag_chat.py -q "..." --k 8 --max-tokens 1024 --ctx 16384
```

The prompt forces answers to stay inside the retrieved sources and to cite
them as `[1]`, `[2]`; every source is printed at the end with its file path.

## Hardware

| Setup | Embedding | Search | Generation |
|---|---|---|---|
| RTX 4070 (12 GB) | bge-m3, fp32, ~30–60 chunks/s | FAISS / numpy, <50 ms/query | Qwen3-4B Q4_K_M via llama.cpp |
| CPU only | e5-small works | BM25 + numpy | `--backend none` |

## Data

- Source: `~/projects/jekyll-ai-blog/_posts/{en,zh}/*.md` (~14.5k posts each)
- Cleaning mirrors `finetune/build_dataset.py` (Liquid tags, kramdown attrs,
  image refs stripped)
- Chunks carry `title`, `heading`, `file`, `lang`, `url` metadata, so answers
  can be traced back to the exact post
- Front matter values such as `type` and `generated` are preserved per chunk

## Notes / next steps

1. Build the full index and record Recall@10 from `eval_rag.py` below
2. Try `BAAI/bge-m3` vs `intfloat/multilingual-e5-small` retrieval quality
3. Add reranking (`BAAI/bge-reranker-v2-m3`) on the top-50 candidates
4. Serve the index behind an HTTP API and point the blog's search box at it

## Results

_To be filled after the first full index build._

| Embedding | Mode | Recall@10 | Recall@1 | MRR |
|---|---|---|---|---|
| bge-m3 | hybrid | | | |
| bge-m3 | dense | | | |
| bge-m3 | bm25 | | | |
