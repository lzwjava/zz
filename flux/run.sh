#!/usr/bin/env bash
#
# run.sh — Generate images with FLUX.1-schnell Q4_0 via stable-diffusion.cpp
#
# Uses the pre-built sd-cli binary (CUDA-enabled) and models in ./models/
#
# Usage:
#   ./run.sh "a serene mountain lake at sunset"
#   ./run.sh "a cat holding a sign" --steps 2
#   ./run.sh                               (uses default prompt)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/models"
SD_CLI="${SCRIPT_DIR}/sd_cpp/build/bin/sd-cli"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"

# Model files
MODEL="${MODEL_DIR}/flux1-schnell-Q4_0.gguf"
VAE="${MODEL_DIR}/ae.safetensors"
CLIP_L="${MODEL_DIR}/clip_l.safetensors"
T5XXL="${MODEL_DIR}/t5xxl_fp16.safetensors"

# --- VRAM-aware defaults ---
# The desktop (Xorg + Chromium + Slack + editors) holds ~2.9 GB of the 12 GB
# RTX 4070, leaving only ~8.9 GB usable for CUDA. FLUX Q4_0 needs 6.4 GB of
# weights + ~2.35 GB compute buffer + overhead -> only 768x768 fits reliably.
# For 1024x1024, free up VRAM first (close Chromium/Slack), then run:
#   WIDTH=1024 HEIGHT=1024 ./run.sh "prompt"
# (Attention memory scales with resolution, so 512x512 also fits comfortably.)
WIDTH="${WIDTH:-768}"
HEIGHT="${HEIGHT:-768}"
STEPS="${STEPS:-4}"
MAX_VRAM="${MAX_VRAM:-10}"
SEED="${SEED:-42}"

DEFAULT_PROMPT="a majestic dragon perched on a medieval castle tower, fantasy art style"

# --- Check prerequisites ---

if [ ! -f "${SD_CLI}" ]; then
    echo "ERROR: sd-cli binary not found at ${SD_CLI}"
    echo "Rebuild with CUDA first:"
    echo "  cd sd_cpp && cmake -S . -B build -DSD_CUDA=ON && cmake --build build -j"
    exit 1
fi

for f in "${MODEL}" "${VAE}" "${CLIP_L}" "${T5XXL}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing model file: $f"
        echo "Run ./download_model.sh first, or check the models/ directory."
        exit 1
    fi
done

# --- Parse args ---

PROMPT="${1:-${DEFAULT_PROMPT}}"
shift 2>/dev/null || true

mkdir -p "${OUTPUT_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${OUTPUT_DIR}/flux_${TIMESTAMP}.png"

echo "========================================"
echo " FLUX.1-schnell Q4_0 — GPU Accelerated"
echo "========================================"
echo " Model:  ${MODEL}"
echo " VAE:    ${VAE}"
echo " Clip:   ${CLIP_L}"
echo " T5:     ${T5XXL}"
echo " Prompt: ${PROMPT}"
echo " Output: ${OUTPUT}"
echo "========================================"
echo ""

# Run with CUDA backend on RTX 4070
#   --backend diffusion=cuda    Flux transformer on GPU
#   --backend vae=cuda          VAE on GPU (with --vae-tiling to fit VRAM)
#   --backend clip=cpu          CLIP on CPU (tiny, negligible impact)
#   --backend t5xxl=cpu         T5 on CPU (re-downloaded from comfy mirror to fix corrupt file)
#   --vae-tiling                Splits VAE decode into tiles (416 MB vs 6.6 GB!)
#   --max-vram ${MAX_VRAM}      VRAM budget (default 10 = headroom for desktop)
#
# Performance breakdown (RTX 4070, 1024x1024, 4 steps):
#   Flux diffusion:  ~12.8s  (GPU)
#   VAE decode:       ~2.7s  (GPU, tiled)
#   Text encoding:    ~0.3s  (CPU)
#   Total:           ~16s    (vs 526s on CPU-only = 33x speedup)
"${SD_CLI}" \
    --diffusion-model "${MODEL}" \
    --vae "${VAE}" \
    --clip_l "${CLIP_L}" \
    --t5xxl "${T5XXL}" \
    --prompt "${PROMPT}" \
    --cfg-scale 1.0 \
    --sampling-method euler \
    --steps "${STEPS}" \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --seed "${SEED}" \
    --output "${OUTPUT}" \
    --backend "diffusion=cuda,clip=cpu,vae=cuda,t5xxl=cpu" \
    --vae-tiling \
    --max-vram "${MAX_VRAM}" \
    --verbose \
    "$@"

echo ""
echo "✓ Image saved to: ${OUTPUT}"
ls -lh "${OUTPUT}"
