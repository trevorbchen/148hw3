"""§4 — Compare full FT, LoRA, and linear probe on RESISC45.

Saves under <output_dir>/:
  - log.csv             per-step / per-epoch metrics
  - metrics.json        final summary (test acc, params, peak mem, time)
  - figures/loss.png    train loss vs step
  - figures/test_acc.png test accuracy vs epoch
  - best.pt

Usage:
    python scripts/finetune_resisc.py --config configs/lora_resisc.yaml \\
        --method lora --rank 8 \\
        --pretrained runs/clip_eurosat/best.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from basics.lora import apply_lora_to_attention
from basics.vit import ViT
from scripts._plot_utils import RunLogger, cosine_with_warmup
from vlm.data import build_resisc45_loaders


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--method", choices=["linear_probe", "lora", "full_ft"], required=True)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--pretrained", type=Path, required=True,
                   help="Path to CLIP-pretrained ViT checkpoint from §3")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


class ViTClassifier(nn.Module):
    """Wraps a ViT backbone with a Linear classifier on the CLS embedding."""

    def __init__(self, vit: ViT, num_classes: int) -> None:
        super().__init__()
        self.vit = vit
        self.head = nn.Linear(vit.d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.vit(x))


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.output_dir is None:
        suffix = f"rank{args.rank}" if args.method == "lora" else "default"
        args.output_dir = Path("runs") / f"resisc_{args.method}_{suffix}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(args.output_dir)
    device = torch.device(args.device)

    # ---------------------------------------------------------------- data
    train_dl, test_dl = build_resisc45_loaders(
        img_size=64,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )

    # ---------------------------------------------------------------- model
    ckpt = torch.load(args.pretrained, map_location="cpu")
    vit_cfg = ckpt["config"]["vit"]
    vit = ViT(**vit_cfg)
    vit.load_state_dict(ckpt["vit"])

    if args.method == "linear_probe":
        for p in vit.parameters():
            p.requires_grad_(False)
    elif args.method == "lora":
        vit = apply_lora_to_attention(vit, rank=args.rank, alpha=args.alpha)
    # full_ft: leave everything trainable

    model = ViTClassifier(vit, num_classes=cfg["num_classes"]).to(device)

    # ---------------------------------------------------------------- optim
    method_lr = cfg["methods"][args.method]["lr"]
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=method_lr,
        weight_decay=cfg["optim"]["weight_decay"],
        betas=tuple(cfg["optim"]["betas"]),
    )

    num_epochs = cfg["train"]["num_epochs"]
    total_steps = len(train_dl) * num_epochs
    warmup_steps = cfg["optim"]["warmup_steps"]

    trainable_params_count = sum(p.numel() for p in trainable)
    total_params_count = sum(p.numel() for p in model.parameters())

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    step = 0
    best_test_acc = -1.0
    t0 = time.time()

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_dl:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            lr_mult = cosine_with_warmup(step, total_steps, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = method_lr * lr_mult
            optimizer.step()

            if step % cfg["train"]["log_every"] == 0:
                logger.log(
                    step=step, epoch=epoch,
                    loss=float(loss.item()), lr=method_lr * lr_mult,
                    grad_norm=float(grad_norm),
                )
                print(f"epoch {epoch} step {step} loss {loss.item():.4f} lr {method_lr * lr_mult:.2e}")
            step += 1

        # ----------- test eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_dl:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                preds = model(images).argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        test_acc = correct / max(total, 1)
        logger.log(step=step, epoch=epoch, test_acc=test_acc)
        print(f"[epoch {epoch}] test acc = {test_acc:.4f}")
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "test_acc": test_acc, "method": args.method},
                args.output_dir / "best.pt",
            )

    elapsed = time.time() - t0
    peak_mem = (
        torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
    )

    # ---------------------------------------------------------------- plots
    logger.line_plot("step", "loss", "loss.png",
                     title=f"RESISC45 train loss ({args.method})",
                     xlabel="step", ylabel="loss")
    logger.line_plot("epoch", "test_acc", "test_acc.png",
                     title=f"RESISC45 test accuracy ({args.method})",
                     xlabel="epoch", ylabel="accuracy")

    metrics = {
        "method": args.method,
        "rank": args.rank if args.method == "lora" else None,
        "best_test_acc": best_test_acc,
        "trainable_params": trainable_params_count,
        "total_params": total_params_count,
        "trainable_fraction": trainable_params_count / total_params_count,
        "peak_memory_mb": peak_mem,
        "wall_time_sec": elapsed,
        "total_steps": step,
    }
    logger.save_metrics(metrics)
    print(f"DONE  method={args.method}  test={best_test_acc:.4f}  "
          f"trainable={trainable_params_count}  peak_mem={peak_mem:.0f}MB  "
          f"time={elapsed:.0f}s  -> {args.output_dir}")


if __name__ == "__main__":
    main()
