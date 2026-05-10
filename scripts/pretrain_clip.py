"""§3 — CLIP-style pretraining on EuroSAT.

Saves under <output_dir>/:
  - log.csv             per-step / per-epoch metrics
  - metrics.json        final summary (best/last val acc, test acc, time, ...)
  - figures/loss.png    train loss vs step
  - figures/lr.png      learning rate schedule
  - figures/val_acc.png val zero-shot accuracy vs epoch
  - figures/logit_scale.png
  - best.pt             best ViT + projection head + logit_scale state

Usage:
    python scripts/pretrain_clip.py --config configs/clip_eurosat.yaml
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

# Allow `python scripts/pretrain_clip.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from basics.text_encoder import FrozenTextEncoder
from basics.vit import ViT
from scripts._plot_utils import RunLogger, cosine_with_warmup
from vlm.clip import ProjectionHeads, clip_loss, init_logit_scale
from vlm.data import EUROSAT_CLASSES, build_eurosat_loaders
from vlm.eval import zeroshot_classification_accuracy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb", action="store_true", help="Log to W&B")
    p.add_argument(
        "--pos-encoding", choices=["learned", "rope1d", "rope2d"], default="learned",
        help="Positional encoding for the ViT (§6 ablation)",
    )
    p.add_argument(
        "--extrapolation-img-size", type=int, default=None,
        help="If set, evaluate on EuroSAT upsampled to this size after training (§6 length extrapolation)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.output_dir is None:
        args.output_dir = Path("runs") / f"clip_eurosat_{args.pos_encoding}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(args.output_dir)
    device = torch.device(args.device)

    if args.wandb:
        import wandb

        wandb.init(project="cs148-hw3-clip", config=cfg, dir=str(args.output_dir))

    # ---------------------------------------------------------------- data
    train_dl, val_dl, test_dl = build_eurosat_loaders(
        img_size=cfg["vit"]["img_size"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )

    # ---------------------------------------------------------------- model
    vit = ViT(**cfg["vit"], pos_encoding=args.pos_encoding).to(device)
    text_encoder = FrozenTextEncoder(cfg["text_encoder"]["model_name"]).to(device)
    projection = ProjectionHeads(
        d_image=cfg["vit"]["d_model"],
        d_text=text_encoder.embedding_dim,
        d_proj=cfg["projection"]["d_proj"],
    ).to(device)
    logit_scale = init_logit_scale().to(device)

    trainable_params = (
        list(vit.parameters()) + list(projection.parameters()) + [logit_scale]
    )

    # ---------------------------------------------------------------- optim
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
        betas=tuple(cfg["optim"]["betas"]),
    )

    num_epochs = cfg["train"]["num_epochs"]
    total_steps = len(train_dl) * num_epochs
    warmup_steps = cfg["optim"]["warmup_steps"]
    base_lr = cfg["optim"]["lr"]

    # ---------------------------------------------------------------- run
    class_prompts = [f"a satellite image of {c}" for c in EUROSAT_CLASSES]
    class_indices = list(range(len(EUROSAT_CLASSES)))

    best_val_acc = -1.0
    step = 0
    t0 = time.time()

    for epoch in range(num_epochs):
        vit.train()
        projection.train()
        for images, captions in train_dl:
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                text_embeds_raw = text_encoder(captions)

            img_proj, txt_proj = projection(vit(images), text_embeds_raw)
            loss = clip_loss(img_proj, txt_proj, logit_scale)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            # Cosine LR schedule
            lr_mult = cosine_with_warmup(step, total_steps, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr * lr_mult
            optimizer.step()
            logit_scale.data.clamp_(max=math.log(100.0))

            if step % cfg["train"]["log_every"] == 0:
                logger.log(
                    step=step,
                    epoch=epoch,
                    loss=float(loss.item()),
                    lr=base_lr * lr_mult,
                    grad_norm=float(grad_norm),
                    logit_scale=float(logit_scale.item()),
                )
                if args.wandb:
                    wandb.log({"train/loss": loss.item(), "train/lr": base_lr * lr_mult,
                               "train/logit_scale": logit_scale.item()}, step=step)
                print(f"epoch {epoch} step {step} loss {loss.item():.4f} "
                      f"lr {base_lr * lr_mult:.2e} logit_scale {logit_scale.item():.3f}")
            step += 1

        # ------------- end-of-epoch eval
        if (epoch + 1) % cfg["train"]["eval_every_epoch"] == 0:
            val_acc = zeroshot_classification_accuracy(
                vit, projection, text_encoder, val_dl,
                class_prompts, class_indices, device,
            )
            logger.log(step=step, epoch=epoch, val_acc=val_acc)
            print(f"[epoch {epoch}] val zero-shot acc = {val_acc:.4f}")
            if args.wandb:
                wandb.log({"val/zeroshot_acc": val_acc}, step=step)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "vit": vit.state_dict(),
                        "projection": projection.state_dict(),
                        "logit_scale": logit_scale.detach().cpu(),
                        "config": cfg,
                        "pos_encoding": args.pos_encoding,
                        "epoch": epoch,
                        "val_acc": val_acc,
                    },
                    args.output_dir / "best.pt",
                )

    elapsed = time.time() - t0

    # Final test accuracy with best checkpoint
    ckpt = torch.load(args.output_dir / "best.pt", map_location=device)
    vit.load_state_dict(ckpt["vit"])
    projection.load_state_dict(ckpt["projection"])
    test_acc = zeroshot_classification_accuracy(
        vit, projection, text_encoder, test_dl,
        class_prompts, class_indices, device,
    )

    # Length-extrapolation eval (§6.1)
    extrap_acc = None
    if args.extrapolation_img_size is not None:
        from vlm.data import build_eurosat_loaders as _b
        _, ext_val_dl, _ = _b(
            img_size=args.extrapolation_img_size,
            batch_size=cfg["train"]["batch_size"],
            num_workers=cfg["train"]["num_workers"],
        )
        extrap_acc = zeroshot_classification_accuracy(
            vit, projection, text_encoder, ext_val_dl,
            class_prompts, class_indices, device,
        )
        print(f"[extrapolation @ {args.extrapolation_img_size}] val acc = {extrap_acc:.4f}")

    # ---------------------------------------------------------------- plots
    logger.line_plot("step", "loss", "loss.png",
                     title="CLIP train loss", xlabel="step", ylabel="loss")
    logger.line_plot("step", "lr", "lr.png",
                     title="Learning rate", xlabel="step", ylabel="lr")
    logger.line_plot("step", "logit_scale", "logit_scale.png",
                     title="Logit scale", xlabel="step", ylabel="ln(1/τ)")
    logger.line_plot("epoch", "val_acc", "val_acc.png",
                     title="Zero-shot val accuracy", xlabel="epoch", ylabel="accuracy")

    metrics = {
        "pos_encoding": args.pos_encoding,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "extrapolation_img_size": args.extrapolation_img_size,
        "extrapolation_val_acc": extrap_acc,
        "best_epoch": int(ckpt["epoch"]),
        "wall_time_sec": elapsed,
        "total_steps": step,
        "trainable_params": sum(p.numel() for p in trainable_params),
    }
    logger.save_metrics(metrics)
    print(f"DONE  best_val={best_val_acc:.4f}  test={test_acc:.4f}  "
          f"time={elapsed:.0f}s  -> {args.output_dir}")


if __name__ == "__main__":
    main()
