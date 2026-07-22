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
#   --backend t5xxl=cpu         T5 on CPU (file may have corrupt tensors)
#   --vae-tiling                Splits VAE decode into tiles (416 MB vs 6.6 GB!)
#   --max-vram 10               Leaves headroom for display etc.
#
# Performance breakdown (RTX 4070, 1024x1024, 4 steps):
#   Flux diffusion:  ~12.8s  (GPU)
#   VAE decode:       ~2.7s  (GPU, tiled)
#   Text encoding:    ~0.3s  (CPU)
#   Total:           ~16s    (vs 526s on CPU-only = 33x speedup)
#
# Note: T5XXL "data offsets out of bounds" error — re-download to fix.
"${SD_CLI}" \
    --diffusion-model "${MODEL}" \
    --vae "${VAE}" \
    --clip_l "${CLIP_L}" \
    --t5xxl "${T5XXL}" \
    --prompt "${PROMPT}" \
    --cfg-scale 1.0 \
    --sampling-method euler \
    --steps 4 \
    --width 1024 \
    --height 1024 \
    --seed 42 \
    --output "${OUTPUT}" \
    --backend "diffusion=cuda,clip=cpu,vae=cuda,t5xxl=cpu" \
    --vae-tiling \
    --max-vram 10 \
    --verbose \
    "$@"

echo ""
echo "✓ Image saved to: ${OUTPUT}"
ls -lh "${OUTPUT}"
