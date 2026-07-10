# Codeparrot-d12 Long-Prompt Evaluation

Model: 286M params (depth=12), step 87000, val_bpb: 0.3723

---
### Sample 1 — PROMPT (INPUT)

```python
class TreeNode:
    """Binary search tree node implementation."""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class BinarySearchTree:
    """Binary search tree with insert, search, and delete operations."""
    def __init__(self):
        self.root = None

    def insert(self, val):
```

### MODEL OUTPUT

```python
  # insert(val)
        if self.root is None:
            self.root = TreeNode(val)
        else:
            self.root.left = self.insert(val)
            self.root.right = self.insert(val)

    def search(self, val):  # search(val)
        if self.root is None:
            return None
        if self.root.left is None:
            return None
        if self.root.right is None:
            return None
        if self.root.left.val == val and self.root.right.val == val:
            return self.root.left.val
        if self.root.left.val == val and self.root.right.val == val:
            return self.root.right.val
        return None

    def delete(self, val):  # delete(val)
        if self.root is None:
            return None
        if self.root.left is None:
            return None
        if self.root.right is None:
            return None
        if self.root.left.val == val and self.root.right.val == val:
            self.root.right = self.root.right.left
            self.root.left = self.root.left.right
            self.root.right = self.root.right.left
            return True

    def __str__(self):
        if self.root is None:
            return "root"
        return "(%s, %s, %s, %s)" % (self.root.left, self.root.right, self.root.val, self.root.right.val)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        if self.root is None:
            return 0
        return len(self.root)

    def __iter__(self):
        if self.root is None:
            return
        yield self.root.val
        yield self.root.left.val
        yield self.root.right.val

    def __getitem__(self, key):
        if self.root is None:
            return None
       
```

---
### Sample 2 — PROMPT (INPUT)

```python
def longest_common_subsequence(text1: str, text2: str) -> int:
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
    
```

### MODEL OUTPUT

```python
 # dp[i][j] represents the LCS length for text1[:i] and text2[:j]
    for i in range(m):
        for j in range(n):
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]) + 1
    return dp[m - 1][n - 1]

def longest_common_subsequence_of(text1: str, text2: str) -> int:
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
    for i in range(m):
        for j in range(n):
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]) + 1
    return dp[m - 1][n - 1]

def longest_common_subsequence_of2(text1: str, text2: str) -> int:
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
    m, n = len(text
```

---
### Sample 3 — PROMPT (INPUT)

```python
import time
import functools


def timer_decorator(func):
    """Decorator that prints the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
```

### MODEL OUTPUT

```python
  # pylint: disable=missing-docstring
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("Execution time: %.2f seconds" % (end - start))
        return result
    return wrapper


def timer(func):
    """Decorator that prints the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # pylint: disable=missing-docstring
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("Execution time: %.2f seconds" % (end - start))
        return result
    return wrapper


def timer_decorator(func):
    """Decorator that prints the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # pylint: disable=missing-docstring
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("Execution time: %.2f seconds" % (end - start))
        return result
    return wrapper


def timer_decorator(func):
    """Decorator that prints the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # pylint: disable=missing-docstring
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("Execution time: %.2f seconds" % (end - start))
        return result
    return wrapper


def timer_decorator(func):
    """Decorator that prints the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # pylint: disable=missing-docstring
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("Execution time: %.
```

---
### Sample 4 — PROMPT (INPUT)

```python
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional


def load_json_data(filepath: str) -> Optional[Dict]:
    """Load JSON data from a file, with proper error handling."""
    try:
        path = Path(filepath)
        if not path.exists():
```

### MODEL OUTPUT

```python
  # pragma: no cover
            raise FileNotFoundError
        with path.open() as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.decoder.JSONDecodeError:
        return None


def load_csv_data(filepath: str) -> List[Dict]:
    """Load CSV data from a file, with proper error handling."""
    with open(filepath) as f:
        reader = csv.reader(f)
        return list(reader)


def load_csv_data_from_json(filepath: str) -> List[Dict]:
    """Load CSV data from a file, with proper error handling."""
    with open(filepath) as f:
        reader = csv.reader(f)
        return list(reader)


def load_csv_data_from_csv(filepath: str) -> List[Dict]:
    """Load CSV data from a file, with proper error handling."""
    with open(filepath) as f:
        reader = csv.reader(f)
        return list(reader)


def load_json_data_from_csv(filepath: str) -> List[Dict]:
    """Load JSON data from a file, with proper error handling."""
    with open(filepath) as f:
        reader = json.load(f)
        return list(reader)


def load_json_data_from_json(filepath: str) -> List[Dict]:
    """Load JSON data from a file, with proper error handling."""
    with open(filepath) as f:
        reader = json.load(f)
        return list(reader)


def load_json_data_from_csv(filepath: str) -> List[Dict]:
    """Load JSON data from a file, with proper error handling."""
    with open(filepath) as f:
        reader = json.load(f)
        return list(reader)


def load_csv_data_from_csv(filepath: str) -> List[Dict]:
    """Load CSV data from a file, with proper error handling."""
    with open(filepath) as f:
        reader = csv.reader(f)
        return list(reader
```

---
### Sample 5 — PROMPT (INPUT)

```python
def is_prime(n: int) -> bool:
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
    while n > 1:
```

### MODEL OUTPUT

```python
  # n is a number
        if n % divisor == 0:
            divisor += 1
        else:
            factors.append(n // divisor)
            n = n % divisor
    return factors


def is_prime_factors(n: int) -> bool:
    """Check if a number is prime factors of a number."""
    factors = prime_factors(n)
    for factor in factors:
        if is_prime(factor):
            return True
    return False


def prime_factors(n: int) -> list:
    """Return the prime factors of a number as a list."""
    factors = prime_factors(n)
    factors = [int(x) for x in factors]
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            factors.append(i)
    return factors


def is_prime_factors(n: int) -> bool:
    """Check if a number is prime factors of a number."""
    factors = prime_factors(n)
    for factor in factors:
        if is_prime(factor):
            return True
    return False


def prime_factors(n: int) -> list:
    """Return the prime factors of a number as a list."""
    factors = prime_factors(n)
    factors = [int(x) for x in factors]
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            factors.append(i)
    return factors


def is_prime_factors(n: int) -> bool:
    """Check if a number is prime factors of a number."""
    factors = prime_factors(n)
    for factor in factors:
        if is_prime(factor):
            return True
    return False


def prime_factors(n: int) -> list:
    """Return the prime factors of a number as a list."""
   
```

---
### Sample 6 — PROMPT (INPUT)

```python
import requests
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
        try:
```

### MODEL OUTPUT

```python
  # pylint: disable=no-value-for-parameter
            return self.session.get(url, params=params, timeout=self.timeout)
        except requests.exceptions.Timeout as e:
            raise TimeoutError(f"Request to {url} timed out: {e}")

    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a POST request with error handling."""
        url = f"{self.base_url}{endpoint}"
        try:  # pylint: disable=no-value-for-parameter
            return self.session.post(url, data=data, timeout=self.timeout)
        except requests.exceptions.Timeout as e:
            raise TimeoutError(f"Request to {url} timed out: {e}")

    def put(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a PUT request with error handling."""
        url = f"{self.base_url}{endpoint}"
        try:  # pylint: disable=no-value-for-parameter
            return self.session.put(url, data=data, timeout=self.timeout)
        except requests.exceptions.Timeout as e:
            raise TimeoutError(f"Request to {url} timed out: {e}")

    def delete(self, endpoint: str) -> Dict[str, Any]:
        """Send a DELETE request with error handling."""
        url = f"{self.base_url}{endpoint}"
        try:  # pylint: disable=no-value-for-parameter
            return self.session.delete(url, timeout=self.timeout)
        except requests.exceptions.Timeout as e:
            raise TimeoutError(f"Request to {url} timed out: {e}")

    def patch(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a PATCH request with error handling."""
        url = f"{self.base_url}{endpoint}"
        try:  # pylint: disable
```

---
### Sample 7 — PROMPT (INPUT)

```python
import pandas as pd
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
    numeric_cols = df.select_dtypes(include=[np.number]).columns
```

### MODEL OUTPUT

```python

    df = df.dropna(subset=numeric_cols)

    # Normalize string columns
    df = df.apply(lambda x: x.lower().strip())

    # Remove outliers
    df = df.dropna(subset=['median', 'mean', 'std', 'var', 'median', 'mean', 'std', 'var', 'median', 'mean', 'std', 'var'])

    # Remove non-NA values
    df = df.dropna(subset=['median', 'mean', 'std', 'var', 'median', 'mean', 'std', 'var'])

    # Remove columns with missing values
    df = df.dropna(subset=['median', 'mean', 'std', 'var', 'median', 'mean', 'std', 'var'])

    # Remove columns with outliers
    df = df.dropna(subset=['median', 'mean', 'std', 'var', 'median', 'mean', 'std', 'var'])

    # Remove columns with non-numeric values
    df = df.dropna(subset=['median', 'mean', 'std', 'var', 'median', 'mean', 'std', 'var'])

    # Remove columns with non-numeric values
    df = df.dropna(subset=['median', 'mean', 'std', 'var', 'median', 'mean', 'std', 'var'])

    # Remove columns with non-numeric values
    df = df.dropna(subset=['median', 'mean', 'std', 'var', 'median', 'mean', 'std', 'var'])

    # Remove columns with non-numeric values
    df = df.dropna(subset=['median', 'mean', 'std', 'var', 'median', 'mean', 'std', 'var'])

    # Remove columns with non-numeric values
    df = df
```

---
### Sample 8 — PROMPT (INPUT)

```python
import asyncio
import aiohttp
from typing import List, Dict, Any


async def fetch_url(session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
    """Fetch a URL asynchronously and return JSON response."""
    async with session.get(url) as response:
```

### MODEL OUTPUT

```python
  # type: ignore
        return await response.json()


async def fetch_json(session: aiohttp.ClientSession, url: str) -> Any:
    """Fetch a JSON response from a URL asynchronously and return JSON."""
    async with session.get(url) as response:  # type: ignore
        return await response.json()


async def fetch_json_async(session: aiohttp.ClientSession, url: str) -> Any:
    """Fetch a JSON response from a URL asynchronously and return JSON."""
    async with session.get(url) as response:  # type: ignore
        return await response.json()


async def fetch_json_async_async(session: aiohttp.ClientSession, url: str) -> Any:
    """Fetch a JSON response from a URL asynchronously and return JSON."""
    async with session.get(url) as response:  # type: ignore
        return await response.json()


async def fetch_json_async_async_async(session: aiohttp.ClientSession, url: str) -> Any:
    """Fetch a JSON response from a URL asynchronously and return JSON."""
    async with session.get(url) as response:  # type: ignore
        return await response.json()


async def fetch_json_async_async_async_async(session: aiohttp.ClientSession, url: str) -> Any:
    """Fetch a JSON response from a URL asynchronously and return JSON."""
    async with session.get(url) as response:  # type: ignore
        return await response.json()


async def fetch_json_async_async_async_async_async(session: aiohttp.ClientSession, url: str) -> Any:
    """Fetch a JSON response from a URL asynchronously and return JSON."""
    async with session.get(url) as response:  # type: ignore
        return await response.json()


async def fetch_json_async_async_async_async_async_async(session: aiohttp.ClientSession, url: str) -> Any:
    """Fetch a JSON response from a URL asynchronous
```

