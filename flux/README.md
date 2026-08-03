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

## Fast local generation (recommended) — stable-diffusion.cpp

The GGUF + sd-cli path (`run.sh`) is the fast route on the RTX 4070:

```bash
./run.sh "a serene mountain lake at sunset"          # 768x768, 4 steps, ~16s
WIDTH=1024 HEIGHT=1024 ./run.sh "prompt"             # needs ~1 GB more free VRAM
```

> **VRAM note**: the desktop (Xorg + Chromium + Slack + editors) holds ~2.9 GB of
> the 12 GB RTX 4070, so only ~8.9 GB is usable by CUDA. FLUX Q4_0 needs 6.4 GB
> of weights + ~2.35 GB compute buffer, so **768×768 is the safe default**.
> For 1024×1024, close Chromium/Slack to free VRAM (worked on Jul 23 with a
> lighter desktop). Env overrides: `WIDTH`, `HEIGHT`, `STEPS`, `MAX_VRAM`, `SEED`.

Also usable from the `ahl` project: `ahl img "prompt"` or
`ahl gen-video ... --provider sdcpp`.

## Notes

- **bfloat16** precision + CPU offloading to balance VRAM usage.
- Schnell is distilled — great results in **just 1–4 inference steps**.
- `guidance_scale=0.0` is correct for the distilled Schnell variant.
- **T5 encoder**: the original `t5xxl_fp16.safetensors` (1.53 GB) was truncated
  ("data offsets out of bounds") and produced CLIP-only conditioning. Replaced
  with the full 9.8 GB file from the comfy mirror (`comfyanonymous/flux_text_encoders`).
  Corrupt file kept as `t5xxl_fp16.safetensors.corrupt`.

## Files

| File | Purpose |
|------|---------|
| `run_flux_schnell.py` | Main script — download & generate |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |
