# ZZ

A personal ML experimentation lab: dataset processing, LLM pretraining, SFT
fine-tuning, GRPO reinforcement learning, and local inference utilities.

It started as scripts for downloading/tokenizing public datasets, and grew into
a full-stack toolkit — from raw CommonCrawl/Wikipedia parquet files all the way
to trained nanochat checkpoints, RL-tuned adapters, RAG over a personal blog,
and local image/chat inference.

## Getting Started

```bash
pip install -r requirements.txt
```

Most workflows use a virtualenv and need a GPU. Entry-point shell scripts assume
the `nanochat` codebase lives at `/mnt/data/nanochat` and this repo is checked
out at `/mnt/data/zz`; adjust paths in the scripts or set the env vars they export.

## Project Layout

```
scripts/            # Dataset download, extraction, tokenization, analysis
prepare_data.sh     # One-shot: download + convert github-code/fineweb-edu/sec-edgar
fineweb-code-sec-gpt.sh   # nanochat d12 (286M) pretraining on mixed data
fineweb-code-sec-gpt/     # SFT data prep + thousands of prompt templates
fineweb-code-sec-gpt-sft/ # SFT checkpoint inference evaluation
eval_sft.sh / eval_sft_extra.sh  # run inference on the SFT checkpoint
eval_grpo_adapter.py       # compare a GRPO LoRA adapter vs. its base model
grpo-rl/            # GRPO RL: teach CoT arithmetic to a base LLM (QLoRA, 12GB)
finetune/           # QLoRA SFT on lzwjava's ~16k notes, export to GGUF
rag/                # RAG over lzwjava's ~29k Jekyll notes (dense + BM25)
flux/               # FLUX.1-schnell local text-to-image inference
codeparrot/         # codeparrot-clean pipeline (download → convert → tokenize → train)
finemath / climbmix / h200   # dataset download/tokenize variants for other hardware
spgispeech/         # fine-tune Whisper on SPGISpeech
config/             # nanoGPT training configs (e.g. GPT-2 124M on github-code)
runpod/             # self-hosted LibreChat (LLM UI) deployment
logs/               # training logs and outputs
datasets/           # downloaded dataset storage
```

## 1. Data Pipeline

Download, extract, and tokenize public datasets for (pre)training.

```bash
# --- Download (FineWeb / FineWeb-Edu) ---
python scripts/download/download_fineweb.py --limit 1000 --output output.txt
python scripts/download/plan_and_download_fineweb.py --target-tokens 5B --dataset fineweb-edu
python scripts/download/plan_and_download_fineweb_gpt3.py --output .../fineweb-edu   # ~100B tokens, resumable
bash scripts/download/wget_fineweb_mirror_*.sh     # via hf-mirror.com (China access)
bash scripts/download/wget_wikimedia_*.sh          # Wikipedia dumps

# --- Other sources ---
python scripts/download/download_github_code.py     # GitHub code
python scripts/download/download_sec_edgar.py       # SEC 10-K filings
python scripts/download/plan_and_download_fineweb.py --target-tokens 5B  # bundled into prepare_data.sh

# --- Extract & tokenize ---
python scripts/extract/extract_fineweb.py
python scripts/extract/extract_fineweb_gpt3.py       # → single train_fineweb.txt
python scripts/extract/tokenize_fineweb_gpt3.py      # → uint16 .npy shards (GPT-2 BPE)
python scripts/extract/extract_wiki.py / extract_wiki_corpus.py
python scripts/extract/rename_fineweb.py             # strip ?download=true suffix

# --- Analysis ---
python scripts/analysis/calculate_duration.py
python scripts/analysis/evaluate.py --file logs/train_log_openweb.txt
```

### One-shot prep for nanochat pretraining

```bash
bash prepare_data.sh                    # download + convert github-code, fineweb-edu, sec-edgar
bash prepare_data.sh --skip-fineweb-dl  # reuse cached fineweb-edu
```

### codeparrot (Python code corpus)

```bash
bash codeparrot/run_pipeline.sh   # download (≈12.7 GB) → convert → tokenize
bash codeparrot/train_codeparrot.sh
```

### Other hardware

`h200/` provides FineWeb-Edu download + tokenize tuned for AMD MI300X / US
acceleration (no mirror/proxy, geo-route aware). `finemath/` and `climbmix/`
are dataset download helpers.

## 2. Pretraining

Trains a d12 (286M param) transformer on mixed data inside the `nanochat`
codebase:

```bash
bash fineweb-code-sec-gpt.sh           # full run (~50k steps, ~16h)
bash fineweb-code-sec-gpt.sh --smoke   # 200-step smoke test
```

`prepare_data.sh` must be run once first. nanoGPT configs (e.g. GPT-2 124M on
14B github-code tokens for a single RTX 4070) live in `config/`.

## 3. Fine-Tuning (SFT)

### nanochat d12 SFT checkpoint — inference evaluation

```bash
bash eval_sft.sh           # general + code reasoning prompts
bash eval_sft_extra.sh     # focused reasoning tests
```

### QLoRA on personal notes (Unsloth, fits 12GB)

```bash
cd finetune
python build_dataset.py    # en+zh posts → JSONL split
python train.py            # QLoRA 4-bit SFT
python eval.py
python export_gguf.py      # → ollama-compatible GGUF
```

### Whisper (SPGISpeech)

```bash
python spgispeech/train_whisper.py                   # whisper-small
python spgispeech/train_whisper.py --model tiny --epochs 1
```

## 4. GRPO Reinforcement Learning

Teach chain-of-thought arithmetic to a **base** (non-instruct) LLM with GRPO,
using 4-bit QLoRA so it fits a single 12GB RTX 4070. The reward is split into
format + correctness signals so you can see which behaviour improves.

```bash
cd grpo-rl
.venv/bin/python train_grpo.py --smoke                # 2-step pipeline check
.venv/bin/python train_grpo.py --model Qwen/Qwen2.5-1.5B --num-epochs 2
.venv/bin/python train_grpo.py --model Qwen/Qwen2.5-3B --grad-accum 8
```

Evaluate an adapter against its *untouched* base model (paired accuracy —
the comparison that actually answers "did RL help?"):

```bash
python eval_grpo_adapter.py --mode indist --n 64 --greedy
python eval_grpo_adapter.py --mode ood   --n 32 --temperature 0.7   # harder held-out problems
python eval_grpo_adapter.py --mode indist --save-json results.json
```

## 5. RAG over Personal Notes

Retrieval-augmented generation over lzwjava's ~29k Jekyll notes (en + zh),
using hybrid retrieval (bge-m3 dense + BM25, RRF fusion) — includes a no-LLM
`search.py` and a grounded `rag_chat.py`. See `rag/README.md` for setup.

## 6. Local Inference & Serving

```bash
# DeepSeek-V2-Lite-Chat (4-bit, fits 12GB VRAM)
python scripts/deepseek/run_lite.py                        # interactive chat
python scripts/deepseek/run_lite.py -p "Your prompt"       # single prompt

# FLUX.1-schnell text-to-image — cd flux, see flux/README.md (gated model)
bash flux/download_model.sh && bash flux/run.sh

# Self-hosted LibreChat UI — see runpod/start_librechat.md
```

## Utilities

- `train_log.sh` — tail / watch / summarize nanochat training logs:
  `bash train_log.sh [--watch | --summary | --last]`
- `requirements.txt` — `gensim`, `mediawiki-dump`, `pandas`, `pyarrow`,
  `huggingface_hub`, `datasets`

## Notes

The `logs/` directory collates training output, inference-quality audits, and
subproject evaluation notes. See `finetune/README.md`, `rag/README.md`, and
`flux/README.md` for component-specific documentation.