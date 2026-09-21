# Adam Adaptive Learning Rate — Empirical Verification

Compares **Adam** against **SGD+momentum**, **RMSProp**, and **AdaGrad** on MNIST
using the same CNN, on a single RTX 4070.

## Environment

Use the **system Python 3.12**, not the Homebrew one (which is Python 3.14):

```bash
/usr/bin/python3.12 --version   # Python 3.12.3
/usr/bin/python3.12 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.11.0+cu130 True
```

Dependencies already present: `torch`, `torchvision`, `matplotlib`.
If `matplotlib` is missing:

```bash
/usr/bin/python3.12 -m pip install --break-system-packages matplotlib
```

## Run

```bash
cd /mnt/data/zz/adam
/usr/bin/python3.12 train_adam_proof.py
```

Options:

```bash
/usr/bin/python3.12 train_adam_proof.py --epochs 20 --batch-size 256
# quick smoke test:
/usr/bin/python3.12 train_adam_proof.py --epochs 2 --max-train-samples 512
```

## Outputs

| File | Description |
|------|-------------|
| `results.json` | Per-optimizer train loss/accuracy curves, final val metrics, wall time |
| `optimizer_comparison.png` | Training loss + accuracy plots |

## What to expect

- **Adam** converges fastest and most stably in early epochs.
- **SGD+momentum** starts slow, catches up late.
- **AdaGrad** slows down later — its accumulated squared-gradient term keeps
  shrinking the effective learning rate (this is the "adaptive LR" being proven).

This mirrors the original paper: *Adam: A Method for Stochastic Optimization*
(https://arxiv.org/pdf/1412.6980).
