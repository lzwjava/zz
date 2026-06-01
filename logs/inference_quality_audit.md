# Inference Quality Audit — GPT-2 760M (step 76K/445K)

**Model:** GPT-2 760M (n_layer=24, n_head=24, n_embd=1536)
**Checkpoint:** out-fineweb-760m/ckpt.pt (step 76,000)
**Val Loss:** 3.1587
**Training:** 76K/445K steps (17%), ~2.46B tokens / 14.58B total
**Hardware:** MI300X 192GB, MFU 108-113%

## Test 1: "The history of artificial intelligence"

**Samples:** 5, max_new_tokens=300, temperature=0.8, top_k=200

**Observations:**
- ✓ Grammatically correct sentences
- ✓ Coherent paragraph structure
- ✓ Relevant topic (AI, computer science)
- ✗ Factually wrong (made-up conferences, dates)
- ✗ Repetitive ("structure of the universe" x3)
- ✗ Loses coherence after ~100 tokens
- ✗ Mixes in unrelated content (Yale-New Haven Teachers Institute)

**Verdict:** Wikipedia-flavored text salad. Looks right, reads wrong.

## Test 2: "def fibonacci(n):"

**Samples:** 3, max_new_tokens=200

**Observations:**
- ✗ Completely fails at code generation
- ✗ Generates ISBN numbers, bibliography entries
- ✗ One sample is pure gibberish ("B2p22:B3:B2p2p3...")

**Verdict:** No code capability at all.

## Test 3: "The capital of France is"

**Samples:** 3, max_new_tokens=200

**Observations:**
- ✗ Factually wrong ("Liguestas, in the province of Aragon")
- ✗ One sample says "Paris" then immediately contradicts itself
- ✗ Generates fake geography (Korea-Yunsu border, Sichuan as capital of China)
- ✓ Grammatically coherent

**Verdict:** Fluent but hallucinates confidently.

## Test 4: "Once upon a time, there was a robot who"

**Samples:** 3, max_new_tokens=200

**Observations:**
- ✓ Reasonable story structure
- ✓ Some creative elements
- ✗ Loses plot quickly, becomes incoherent
- ✗ Injects Bible references randomly
- ✗ One sample veers into video game essay

**Verdict:** Has story-like structure but no real narrative logic.

## Overall Assessment

### What it CAN do:
- Generate grammatically correct English
- Maintain topic for ~50-100 tokens
- Produce plausible-looking text structure (lists, paragraphs)
- Mimic writing styles (Wikipedia, academic, narrative)

### What it CANNOT do:
- Factual accuracy (hallucinates confidently)
- Code generation (completely broken)
- Long-range coherence (loses thread after ~100 tokens)
- Reasoning or logic
- Self-consistency (contradicts itself)

### Comparison:
- Better than random (obviously)
- Worse than GPT-2 1.5B (2019) which had similar loss
- Comparable to a very early autocomplete
- NOT usable for any production task

### Bottom line:
At val_loss=3.16, the model has learned language statistics but not language understanding. It's a "stochastic parrot" in the most literal sense — it produces plausible-looking text without any comprehension of meaning.

To get usable quality, you'd need:
- val_loss < 2.5 (roughly GPT-2 level)
- Which means more training or a larger model
- Or: use the checkpoint as a base for fine-tuning

## Files

- inference_760m_step76k_ai_history.txt — AI history prompt samples
- inference_760m_step76k_code.txt — Code generation samples
- inference_760m_step76k_facts.txt — Factual knowledge samples
- inference_760m_step76k_story.txt — Creative writing samples
