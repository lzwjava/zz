#!/usr/bin/env python3
"""Prepare SFT data for the code-sec-fineweb model.

Creates a JSONL file with chat conversations for supervised fine-tuning.
Each line: [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]

Usage:
  python prepare_sft_data.py                          # default 500 examples
  python prepare_sft_data.py --num 2000 --out all.jsonl

Then run SFT:
  python -m scripts.chat_sft --run=code-sec-sft --model-step=50000 --model-tag=d12
"""
import argparse, json, random

parser = argparse.ArgumentParser()
parser.add_argument("--num", type=int, default=500)
parser.add_argument("--out", default="/mnt/data/zz/datasets/nanochat-mixed/sft_code_sec.jsonl")
args = parser.parse_args()

conversations = []

# === CODE EXAMPLES (Python, Go, SQL, React) ===
def py_fib():
    return [
        {"role": "user", "content": "Write a Python function that returns the nth Fibonacci number using recursion."},
        {"role": "assistant", "content": "def fibonacci(n: int) -> int:\n    \"\"\"Return the nth Fibonacci number recursively.\"\"\"\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)"},
    ]

def py_quicksort():
    return [
        {"role": "user", "content": "Implement quicksort in Python with a random pivot."},
        {"role": "assistant", "content": "import random\n\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = random.choice(arr)\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)"},
    ]

def py_merge_sort():
    return [
        {"role": "user", "content": "Write a Python function that implements merge sort with type hints."},
        {"role": "assistant", "content": "from typing import List\n\ndef merge_sort(arr: List[int]) -> List[int]:\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left: List[int], right: List[int]) -> List[int]:\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    result.extend(left[i:])\n    result.extend(right[j:])\n    return result"},
    ]

def py_api():
    return [
        {"role": "user", "content": "Build a FastAPI endpoint that returns server status as JSON with uptime and request count."},
        {"role": "assistant", "content": "from fastapi import FastAPI\nimport time\n\napp = FastAPI()\nstart_time = time.time()\nrequest_count = 0\n\n@app.get(\"/status\")\nasync def get_status():\n    global request_count\n    request_count += 1\n    return {\n        \"uptime_seconds\": time.time() - start_time,\n        \"request_count\": request_count,\n        \"status\": \"healthy\"\n    }"},
    ]

def go_server():
    return [
        {"role": "user", "content": "Write a Go HTTP server with a /health endpoint that responds with JSON."},
        {"role": "assistant", "content": 'package main\n\nimport (\n\t"encoding/json"\n\t"log"\n\t"net/http"\n)\n\ntype HealthResponse struct {\n\tStatus string `json:"status"`\n}\n\nfunc healthHandler(w http.ResponseWriter, r *http.Request) {\n\tw.Header().Set("Content-Type", "application/json")\n\tjson.NewEncoder(w).Encode(HealthResponse{Status: "ok"})\n}\n\nfunc main() {\n\thttp.HandleFunc("/health", healthHandler)\n\tlog.Fatal(http.ListenAndServe(":8080", nil))\n}'},
    ]

def sql_query():
    return [
        {"role": "user", "content": "Write a SQL query to find the top 3 customers by total spending in 2024. Tables: customers(id, name, email), orders(id, customer_id, order_date, total_amount). Exclude cancelled orders."},
        {"role": "assistant", "content": "SELECT c.name, c.email, SUM(o.total_amount) as total_spent\nFROM customers c\nJOIN orders o ON c.id = o.customer_id\nWHERE o.order_date >= '2024-01-01'\n  AND o.order_date < '2025-01-01'\n  AND o.status != 'cancelled'\nGROUP BY c.id, c.name, c.email\nORDER BY total_spent DESC\nLIMIT 3;"},
    ]

def react_component():
    return [
        {"role": "user", "content": "Create a React component that fetches and displays data from an API with loading and error states."},
        {"role": "assistant", "content": "import React, { useState, useEffect } from 'react';\n\nfunction DataFetcher({ url }) {\n  const [data, setData] = useState(null);\n  const [loading, setLoading] = useState(true);\n  const [error, setError] = useState(null);\n\n  useEffect(() => {\n    fetch(url)\n      .then(res => { if (!res.ok) throw new Error('Network error'); return res.json(); })\n      .then(json => { setData(json); setLoading(false); })\n      .catch(err => { setError(err.message); setLoading(false); });\n  }, [url]);\n\n  if (loading) return <div className=\"spinner\">Loading...</div>;\n  if (error) return <div className=\"error\">Error: {error}</div>;\n  return <pre>{JSON.stringify(data, null, 2)}</pre>;\n}\n\nexport default DataFetcher;"},
    ]

# === SEC EXAMPLES ===
def sec_analysis():
    return [
        {"role": "user", "content": "A company has Current Ratio 1.8, D/E 0.45, ROE 16.2%, Revenue growth 18% YoY. Industry averages are 1.5, 0.6, 12%, 10%. Assess the company."},
        {"role": "assistant", "content": "The company outperforms industry averages across all metrics:\n- Liquidity: Current ratio 1.8 vs 1.5 indicates stronger short-term solvency.\n- Leverage: D/E of 0.45 vs 0.6 shows more conservative debt usage, reducing financial risk.\n- Profitability: ROE of 16.2% vs 12% suggests superior capital efficiency.\n- Growth: Revenue growth of 18% vs 10% nearly doubles the industry rate.\nRisks: The combination of high growth and strong margins may attract competition. Recommend monitoring cash conversion cycle and market share trends."},
    ]

def sec_risk_factor():
    return [
        {"role": "user", "content": "Draft a risk factor about AI regulatory compliance for a fintech company's 10-K filing."},
        {"role": "assistant", "content": "Item 1A. Risk Factors\n\nOur AI-powered financial products are subject to evolving regulatory frameworks. Changes in laws governing artificial intelligence, machine learning, and automated decision-making could require us to modify our algorithms, increase compliance costs, or limit our product offerings. Regulatory bodies in jurisdictions where we operate may impose new requirements around model transparency, bias testing, or customer disclosure that could materially adversely affect our business, financial condition, and results of operations."},
    ]

def sec_financial():
    return [
        {"role": "user", "content": "Given revenue of $3.2B (up 14% YoY), net income $487M, EPS $2.43, compute net profit margin and P/E ratio."},
        {"role": "assistant", "content": "Net Profit Margin = Net Income / Revenue = $487M / $3,200M = 15.2%\n\nP/E Ratio = Share Price / EPS. If the stock trades at $58.32, then:\nP/E = $58.32 / $2.43 = 24.0x\n\nThis P/E of 24x is reasonable for a company with 14% revenue growth and 15.2% net margins. The PEG ratio would be 24/14 = 1.7, suggesting the stock is fairly valued relative to its growth rate."},
    ]

def sec_mdna():
    return [
        {"role": "user", "content": "Write an MD&A section discussing a 22% increase in subscription revenue and 5% decline in licensing revenue."},
        {"role": "assistant", "content": "Results of Operations\n\nRevenue for the fiscal year increased 14% to $3.2 billion, driven by a 22% increase in subscription revenue to $2.1 billion, partially offset by a 5% decline in licensing revenue to $1.1 billion. The subscription growth reflects increased customer adoption of our cloud platform, with 12,000 net new enterprise customers added during the year. The licensing decline was expected as we continue our strategic transition to subscription-based offerings. We expect this trend to continue, with subscription revenue representing an increasing proportion of total revenue."},
    ]

# === GENERAL EXAMPLES ===
def explain_compound():
    return [
        {"role": "user", "content": "Explain compound interest and why it's powerful. Show $10,000 at 7% over 30 years."},
        {"role": "assistant", "content": "Compound interest means earning interest on your interest. After 30 years at 7%:\n\nCompound: $10,000 x (1.07)^30 = $76,123\nSimple: $10,000 x (1 + 0.07 x 30) = $31,000\n\nCompound earns 2.5x more! The key is time — the effect accelerates in later years because the interest base keeps growing."},
    ]

def explain_photosynthesis():
    return [
        {"role": "user", "content": "Explain photosynthesis simply."},
        {"role": "assistant", "content": "Photosynthesis converts sunlight into chemical energy. Plants use chlorophyll to absorb light, split water into oxygen and hydrogen, and fix CO2 into glucose via the Calvin cycle. Key equation: 6CO2 + 6H2O + sunlight → C6H12O6 + 6O2. Rubisco is the enzyme that fixes CO2, making it one of the most important enzymes on Earth."},
    ]

def explain_water_cycle():
    return [
        {"role": "user", "content": "Describe the water cycle."},
        {"role": "assistant", "content": "The water cycle: (1) Evaporation — sun heats water in oceans/lakes, turning it to vapor. (2) Condensation — vapor rises, cools, forms clouds. (3) Precipitation — water falls as rain/snow. (4) Collection — water returns to oceans/lakes. (5) Transpiration — plants release water vapor through leaves. This cycle is Earth's natural water purification and distribution system."},
    ]

# All conversation generators
generators = [
    py_fib, py_quicksort, py_merge_sort, py_api, go_server, sql_query, react_component,
    sec_analysis, sec_risk_factor, sec_financial, sec_mdna,
    explain_compound, explain_photosynthesis, explain_water_cycle,
]

random.seed(42)
for i in range(args.num):
    gen = random.choice(generators)
    conv = gen()
    conversations.append(conv)

with open(args.out, "w") as f:
    for conv in conversations:
        f.write(json.dumps(conv) + "\n")

print(f"Wrote {len(conversations)} conversations to {args.out}")