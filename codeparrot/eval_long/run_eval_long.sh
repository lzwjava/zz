#!/bin/bash
# Long-prompt evaluation for codeparrot-d12 base model
# Tests the model's ability to handle longer context and generate coherent code

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

NANOCHAT_DIR="/mnt/data/nanochat"
cd "$NANOCHAT_DIR"
source .venv/bin/activate

echo "=== Codeparrot-d12: Long-Prompt Evaluation ==="
echo ""

python3 -c "
import sys, os
sys.path.insert(0, '$NANOCHAT_DIR')
from nanochat.checkpoint_manager import load_model
from nanochat.tokenizer import HuggingFaceTokenizer
from nanochat.engine import Engine
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

model, tokenizer, meta = load_model('base', device, phase='eval', model_tag='d12')
print(f'Loaded model step {meta[\"step\"]}, val_bpb: {meta[\"val_bpb\"]:.4f}')
engine = Engine(model, tokenizer)

# Longer, more structured prompts
prompts = [
    # 1) Class with docstring, type hints, and multiple methods
    '''class TreeNode:
    \"\"\"Binary search tree node implementation.\"\"\"
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class BinarySearchTree:
    \"\"\"Binary search tree with insert, search, and delete operations.\"\"\"
    def __init__(self):
        self.root = None

    def insert(self, val):''',

    # 2) Detailed docstring with algorithm description
    '''def longest_common_subsequence(text1: str, text2: str) -> int:
    \"\"\"
    Compute the length of the longest common subsequence between two strings.
    
    Uses dynamic programming with a 2D table where dp[i][j] represents
    the LCS length for text1[:i] and text2[:j].
    
    Args:
        text1: First input string
        text2: Second input string
    
    Returns:
        Integer length of the longest common subsequence
    \"\"\"
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    ''',

    # 3) Decorator + context manager
    '''import time
import functools


def timer_decorator(func):
    \"\"\"Decorator that prints the execution time of a function.\"\"\"
    @functools.wraps(func)
    def wrapper(*args, **kwargs):''',

    # 4) File I/O with error handling
    '''import json
import csv
from pathlib import Path
from typing import List, Dict, Optional


def load_json_data(filepath: str) -> Optional[Dict]:
    \"\"\"Load JSON data from a file, with proper error handling.\"\"\"
    try:
        path = Path(filepath)
        if not path.exists():''',

    # 5) Multi-function module with helper functions
    '''def is_prime(n: int) -> bool:
    \"\"\"Check if a number is prime.\"\"\"
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


def prime_factors(n: int) -> list:
    \"\"\"Return the prime factors of a number as a list.\"\"\"
    factors = []
    divisor = 2
    while n > 1:''',

    # 6) Web API client pattern
    '''import requests
from typing import Optional, Dict, Any


class APIClient:
    \"\"\"Simple REST API client with retry logic.\"\"\"
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip(\"/\")
        self.session = requests.Session()
        self.session.headers.update({
            \"User-Agent\": \"APIClient/1.0\",
            \"Accept\": \"application/json\"
        })
        self.timeout = timeout
    
    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        \"\"\"Send a GET request with error handling.\"\"\"
        url = f\"{self.base_url}{endpoint}\"
        try:''',

    # 7) Data processing pipeline
    '''import pandas as pd
import numpy as np
from typing import List, Dict


def clean_and_transform(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"
    Clean a dataframe by:
    1. Removing duplicate rows
    2. Filling missing numeric values with median
    3. Normalizing string columns (strip whitespace, lowercase)
    4. Removing outliers using IQR method
    \"\"\"
    df = df.copy()
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns''',

    # 8) Async pattern
    '''import asyncio
import aiohttp
from typing import List, Dict, Any


async def fetch_url(session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
    \"\"\"Fetch a URL asynchronously and return JSON response.\"\"\"
    async with session.get(url) as response:''',
]

for prompt in prompts:
    tokens = tokenizer(prompt, prepend='<|bos|>')
    prompt_len = len(tokens)  # number of prompt tokens (BOS + prompt)
    sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=384, temperature=0.3)
    generated_tokens = sample[0][prompt_len:]
    generated_str = tokenizer.decode(generated_tokens)
    print('=' * 70)
    print('=== PROMPT (INPUT) ===')
    print(prompt)
    print('--- MODEL OUTPUT ---')
    print(generated_str)
" 2>&1 | tee "$RESULTS_DIR/long_prompt_results.log"

echo ""
echo "=== Done ==="
echo "Results: $RESULTS_DIR/long_prompt_results.log"