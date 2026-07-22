#!/usr/bin/env bash
#
# run.sh — Generate images with FLUX.1-schnell Q4_0 via stable-diffusion.cpp
#
# Uses the pre-built sd-cli binary and models already in ./models/
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
    echo "Build it first or check the path."
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
echo " FLUX.1-schnell Q4_0 — Image Generator"
echo "========================================"
echo " Model:  ${MODEL}"
echo " VAE:    ${VAE}"
echo " Clip:   ${CLIP_L}"
echo " T5:     ${T5XXL}"
echo " Prompt: ${PROMPT}"
echo " Output: ${OUTPUT}"
echo "========================================"
echo ""

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
    --clip-on-cpu \
    --verbose \
    "$@"

echo ""
echo "✓ Image saved to: ${OUTPUT}"
ls -lh "${OUTPUT}"
