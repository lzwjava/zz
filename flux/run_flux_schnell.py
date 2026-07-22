#!/usr/bin/env python3
"""
FLUX.1-schnell — Download & Run
================================
Downloads the gated model (requires HF token) and generates images from text.

Usage:
    # Use HF token from env or --token
    python run_flux_schnell.py --token hf_xxxx

    # Or set env var before running
    export HUGGING_FACE_HUB_TOKEN=hf_xxxx
    python run_flux_schnell.py "a serene mountain lake at sunset"

    # Multiple prompts
    python run_flux_schnell.py "prompt1" "prompt2"
"""

import argparse
import os
import re
import sys

import torch
from diffusers import FluxPipeline

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
MODEL_ID = "black-forest-labs/FLUX.1-schnell"

DEFAULT_PROMPTS = [
    "A cat holding a sign that says 'hello world'",
    "A cyberpunk street at night, neon reflections on wet pavement, cinematic lighting",
    "A majestic dragon perched on a medieval castle tower, fantasy art style",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="FLUX.1-schnell — Download & Generate Images"
    )
    parser.add_argument(
        "prompts",
        nargs="*",
        help="Text prompt(s) for image generation. Uses defaults if none given.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face access token (or set HUGGING_FACE_HUB_TOKEN env var)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="Number of inference steps (1-4, default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed (auto-incremented for each prompt). Default: 0",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    return parser.parse_args()


def sanitize_filename(s: str, max_len: int = 80) -> str:
    """Turn a prompt into a safe filename fragment."""
    s = re.sub(r"[^\w\s-]", "_", s)
    s = re.sub(r"\s+", "_", s.strip())
    return s[:max_len]


def generate_image(pipe, prompt: str, seed: int, index: int, out_dir: str):
    """Generate a single image from a prompt."""
    print(f"\n[{index}] Prompt: {prompt}")
    print(f"    Seed:   {seed}")

    image = pipe(
        prompt,
        guidance_scale=0.0,          # Schnell uses distilled CFG → 0.0
        num_inference_steps=4,        # 1–4 steps (distilled for speed)
        max_sequence_length=256,      # max token length
        generator=torch.Generator("cpu").manual_seed(seed),
    ).images[0]

    os.makedirs(out_dir, exist_ok=True)
    safe = sanitize_filename(prompt)
    path = os.path.join(out_dir, f"flux_schnell_{index:03d}_{safe}.png")
    image.save(path)
    print(f"    Saved:  {path}")
    return path


def main():
    args = parse_args()

    # Resolve token: CLI arg > env var
    token = args.token or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("=" * 60)
        print("ERROR: No Hugging Face token found!")
        print()
        print("This model is gated. You must:")
        print("  1. Visit https://huggingface.co/black-forest-labs/FLUX.1-schnell")
        print("  2. Click 'Agree and access repository' to accept the terms")
        print("  3. Create a token at https://huggingface.co/settings/tokens")
        print()
        print("Then pass it via:")
        print("  --token hf_xxxx")
        print("  # or")
        print("  export HUGGING_FACE_HUB_TOKEN=hf_xxxx")
        print("=" * 60)
        sys.exit(1)

    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
    base_seed = args.seed if args.seed is not None else 0

    print("=" * 60)
    print("FLUX.1-schnell — Loading model...")
    print(f"    Model:  {MODEL_ID}")
    print(f"    Token:  {'✓ provided (starts with ' + token[:4] + '...)' if token else '✗ missing'}")
    print("=" * 60)

    # Load the pipeline with authentication
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        token=token,
    )
    pipe.enable_model_cpu_offload()  # offload unused parts to CPU → saves VRAM

    print(f"    Device: {pipe.device}")
    print(f"    Model loaded successfully!\n")
    print("-" * 60)

    paths = []
    for i, prompt in enumerate(prompts, 1):
        path = generate_image(pipe, prompt, seed=base_seed + i, index=i, out_dir=args.output)
        paths.append(path)

    print("\n" + "=" * 60)
    print(f"Done! {len(paths)} image(s) saved to: {args.output}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
