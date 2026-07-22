#!/usr/bin/env bash
#
# download_model.sh — Download FLUX.1-schnell GGUF (Q4_0) model
#
# Downloads from Hugging Face:
#   https://huggingface.co/aifoundry-org/FLUX.1-schnell-Quantized
#
# Usage:
#   chmod +x download_model.sh
#   ./download_model.sh
#
# The model file is saved to: ./models/flux1-schnell-Q4_0.gguf

set -euo pipefail

MODEL_DIR="$(cd "$(dirname "$0")" && pwd)/models"
MODEL_FILE="${MODEL_DIR}/flux1-schnell-Q4_0.gguf"
URL="https://huggingface.co/aifoundry-org/FLUX.1-schnell-Quantized/resolve/main/flux1-schnell-Q4_0.gguf"

mkdir -p "${MODEL_DIR}"

if [ -f "${MODEL_FILE}" ]; then
    echo "✓ Model already exists at: ${MODEL_FILE}"
    echo "  Delete it first if you want to re-download."
    exit 0
fi

echo "Downloading FLUX.1-schnell Q4_0 GGUF..."
echo "  From: ${URL}"
echo "  To:   ${MODEL_FILE}"
echo ""

# Prefer curl, fallback to wget
if command -v curl &>/dev/null; then
    curl -L -o "${MODEL_FILE}" "${URL}"
elif command -v wget &>/dev/null; then
    wget -O "${MODEL_FILE}" "${URL}"
else
    echo "ERROR: Neither curl nor wget is installed."
    exit 1
fi

echo ""
echo "✓ Download complete!"
ls -lh "${MODEL_FILE}"
