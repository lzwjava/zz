# lzwjava notes fine-tuning

Fine-tune a model on lzwjava's ~16k notes (en + zh) from jekyll-ai-blog.

## Quick start

```bash
# 1. Build dataset
pip install python-frontmatter tiktoken
python build_dataset.py

# 2. Install Unsloth (CUDA / RTX 4070)
pip install unsloth

# 3. Train (QLoRA 4-bit, fits in 12GB)
python train.py

# 4. Eval
python eval.py

# 5. Export GGUF for ollama
python export_gguf.py
```

## Files

| File | Purpose |
|---|---|
| `build_dataset.py` | Extract en+zh posts → JSONL train/eval split |
| `train.py` | QLoRA SFT with Unsloth (configurable) |
| `eval.py` | Compare fine-tuned vs reference on held-out titles |
| `export_gguf.py` | Export merged model to GGUF for ollama/llama.cpp |
| `notes_sft_train.jsonl` | Training data (generated) |
| `notes_sft_eval.jsonl` | Eval data, 200 examples (generated) |

## Train options

```bash
# Default: Qwen3-8B, 4-bit QLoRA, r=32, 2 epochs
python train.py

# Smaller model (faster, less VRAM)
python train.py --model unsloth/Qwen3-4B

# More epochs
python train.py --epochs 3

# Larger batch (if VRAM allows)
python train.py --batch-size 4 --grad-accum 4

# 16-bit LoRA (needs >24GB VRAM)
python train.py --no-4bit --model unsloth/Qwen3-8B
```

## Hardware

| Setup | Model | Method | Time |
|---|---|---|---|
| RTX 4070 (12GB) | Qwen3-4B/8B | QLoRA 4-bit | ~2-4 hours |
| MI300X (192GB) | Qwen3-32B/70B | LoRA 16-bit | <1 hour |

## Data

- 8055 English posts + 8052 Chinese posts = ~16k examples
- Format: `{question: title, answer: body}`
- Train/eval split: 80/20 (200 held out for eval)
- Includes AI-generated posts (no filtering)

## Next steps

1. Run `build_dataset.py` to see token counts
2. Train with default settings on the 4070
3. Eval — check if the "lzwjava voice" emerges
4. If good, try a larger model on MI300X
5. DPO: take edited answers vs raw model answers for RLHF
