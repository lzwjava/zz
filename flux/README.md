# FLUX.1-schnell — Local Inference

[FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) is a 12B parameter text-to-image model by Black Forest Labs. This setup downloads and runs it locally.

## Requirements

- **Python 3.10+**
- **~24 GB VRAM** (GPU recommended) — CPU offloading works on less VRAM but is slower
- **Hugging Face access token** (the model is gated)

## Get Access

FLUX.1-schnell is **gated**. You must:

1. Visit https://huggingface.co/black-forest-labs/FLUX.1-schnell
2. Click **"Agree and access repository"** to accept the terms
3. Create a token at https://huggingface.co/settings/tokens

## Setup

```bash
cd flux

# 1) Create a virtual env (optional but recommended)
python -m venv .venv && source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt
```

> ⚠️ The first run downloads ~24 GB of model weights — this will take a while.

## Usage

```bash
# Pass token via --token
python run_flux_schnell.py --token hf_xxxxx

# Or via env var
export HUGGING_FACE_HUB_TOKEN=hf_xxxxx
python run_flux_schnell.py

# Custom prompts
python run_flux_schnell.py --token hf_xxxxx "a serene mountain lake at sunset"

# Multiple prompts
python run_flux_schnell.py --token hf_xxxxx "prompt 1" "prompt 2"

# Custom steps (1–4) and seed
python run_flux_schnell.py --token hf_xxxxx --steps 2 --seed 42 "neon cityscape"
```

Output images are saved to `./outputs/`.

## Notes

- **bfloat16** precision + CPU offloading to balance VRAM usage.
- Schnell is distilled — great results in **just 1–4 inference steps**.
- `guidance_scale=0.0` is correct for the distilled Schnell variant.

## Files

| File | Purpose |
|------|---------|
| `run_flux_schnell.py` | Main script — download & generate |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |
