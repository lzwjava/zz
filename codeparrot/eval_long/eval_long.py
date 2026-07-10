#!/usr/bin/env python3
"""Long-prompt evaluation for codeparrot-d12 base model (286M params).
Output is Markdown with python syntax-highlighted code blocks."""

import sys
import os
import torch

sys.path.insert(0, "/mnt/data/nanochat")
from nanochat.checkpoint_manager import load_model
from nanochat.tokenizer import HuggingFaceTokenizer
from nanochat.engine import Engine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model, tokenizer, meta = load_model("base", device, phase="eval", model_tag="d12")
print(f"Loaded model step {meta['step']}, val_bpb: {meta['val_bpb']:.4f}")
engine = Engine(model, tokenizer)

prompts = [
    # 1) BST
    '''class TreeNode:
    """Binary search tree node implementation."""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class BinarySearchTree:
    """Binary search tree with insert, search, and delete operations."""
    def __init__(self):
        self.root = None

    def insert(self, val):''',

    # 2) LCS
    '''def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    Compute the length of the longest common subsequence between two strings.

    Uses dynamic programming with a 2D table where dp[i][j] represents
    the LCS length for text1[:i] and text2[:j].

    Args:
        text1: First input string
        text2: Second input string

    Returns:
        Integer length of the longest common subsequence
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    ''',

    # 3) Decorator
    '''import time
import functools


def timer_decorator(func):
    """Decorator that prints the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):''',

    # 4) File I/O
    '''import json
import csv
from pathlib import Path
from typing import List, Dict, Optional


def load_json_data(filepath: str) -> Optional[Dict]:
    """Load JSON data from a file, with proper error handling."""
    try:
        path = Path(filepath)
        if not path.exists():''',

    # 5) Prime factors
    '''def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


def prime_factors(n: int) -> list:
    """Return the prime factors of a number as a list."""
    factors = []
    divisor = 2
    while n > 1:''',

    # 6) API client
    '''import requests
from typing import Optional, Dict, Any


class APIClient:
    """Simple REST API client with retry logic."""

    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "APIClient/1.0",
            "Accept": "application/json"
        })
        self.timeout = timeout

    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Send a GET request with error handling."""
        url = f"{self.base_url}{endpoint}"
        try:''',

    # 7) Data pipeline
    '''import pandas as pd
import numpy as np
from typing import List, Dict


def clean_and_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a dataframe by:
    1. Removing duplicate rows
    2. Filling missing numeric values with median
    3. Normalizing string columns (strip whitespace, lowercase)
    4. Removing outliers using IQR method
    """
    df = df.copy()

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns''',

    # 8) Async
    '''import asyncio
import aiohttp
from typing import List, Dict, Any


async def fetch_url(session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
    """Fetch a URL asynchronously and return JSON response."""
    async with session.get(url) as response:''',
]

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
out_path = os.path.join(out_dir, "long_prompt_results.md")

with open(out_path, "w") as f:
    f.write(f"# Codeparrot-d12 Long-Prompt Evaluation\n\n")
    f.write(f"Model: 286M params (depth=12), step {meta['step']}, val_bpb: {meta['val_bpb']:.4f}\n\n")

    for i, prompt in enumerate(prompts, 1):
        tokens = tokenizer(prompt, prepend="<|bos|>")
        prompt_len = len(tokens)
        sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=384, temperature=0.3)
        generated_tokens = sample[0][prompt_len:]
        generated_str = tokenizer.decode(generated_tokens)

        block = f"""---
### Sample {i} — PROMPT (INPUT)

```python
{prompt}
```

### MODEL OUTPUT

```python
{generated_str}
```

"""
        f.write(block)
        print(block, end="")  # also print to stdout

print(f"\nResults written to: {out_path}")