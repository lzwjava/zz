"""
Empirically verify Adam's adaptive learning rate behavior on a single GPU.

Compares Adam vs SGD, RMSProp, and AdaGrad on MNIST using the same CNN.
Produces:
  - results.json               (loss + accuracy curves per optimizer)
  - optimizer_comparison.png   (training loss + validation accuracy plots)

Run:
    python train_adam_proof.py            # single GPU (RTX 4070)
    python train_adam_proof.py --epochs 20 --batch-size 256

Expected outcome (reproducing the Adam paper's behavior):
  - Adam converges fastest and most stably in early epochs.
  - SGD + momentum catches up later.
  - AdaGrad slows down later because its accumulated gradient squared
    term keeps shrinking the effective learning rate.
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import matplotlib

matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt


# ── Simple CNN model ───────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


# ── Train / evaluate one epoch ─────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (model(imgs).argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        loss = criterion(model(imgs), labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (model(imgs).argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


# ── Optimizer factories (keep learning rates fair & conventional) ─────────────
def optimizer_factories():
    return {
        "Adam": lambda p: torch.optim.Adam(p, lr=1e-3, betas=(0.9, 0.999)),
        "SGD": lambda p: torch.optim.SGD(p, lr=1e-2, momentum=0.9),
        "RMSProp": lambda p: torch.optim.RMSprop(p, lr=1e-3),
        "AdaGrad": lambda p: torch.optim.Adagrad(p, lr=1e-2),
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="optional: cap dataset size for a quick smoke test")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Dataset
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=tf)
    val_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=tf)

    if args.max_train_samples:
        train_ds = Subset(train_ds, range(args.max_train_samples))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    results = {}

    for name, opt_fn in optimizer_factories().items():
        torch.manual_seed(args.seed)  # identical init for every optimizer
        model = SimpleCNN().to(device)
        optimizer = opt_fn(model.parameters())

        losses, accs = [], []
        start = time.time()
        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device)
            losses.append(train_loss)
            accs.append(train_acc)
            print(f"[{name:>7}] Epoch {epoch + 1:02d}/{args.epochs} | "
                  f"loss {train_loss:.4f} | acc {train_acc * 100:.2f}%")

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        results[name] = {
            "train_loss": losses,
            "train_acc": accs,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "wall_time_s": round(time.time() - start, 2),
        }
        print(f"[{name:>7}] Final validation | loss {val_loss:.4f} | "
              f"acc {val_acc * 100:.2f}% | {results[name]['wall_time_s']}s\n")

    # Save + plot
    out_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, r in results.items():
        axes[0].plot(r["train_loss"], label=name, linewidth=2)
        axes[1].plot([a * 100 for a in r["train_acc"]], label=name, linewidth=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss (lower = faster convergence)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Training Accuracy (%)")
    axes[1].set_title("Training Accuracy")
    for ax in axes:
        ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle("Adam vs SGD / RMSProp / AdaGrad (1×RTX 4070)", fontsize=13)
    fig.tight_layout()
    out_png = os.path.join(out_dir, "optimizer_comparison.png")
    fig.savefig(out_png, dpi=150)
    print(f"Saved: {out_png}")
    print(f"Saved: {os.path.join(out_dir, 'results.json')}")


if __name__ == "__main__":
    main()
