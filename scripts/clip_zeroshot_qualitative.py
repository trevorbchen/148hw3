"""§3.3 — Zero-shot qualitative analysis on EuroSAT.

Picks 5 correctly classified and 5 incorrectly classified val examples and
saves images plus top-3 predicted classes.

Outputs under <output_dir>/:
  - examples.jsonl                          one row per saved example
  - images/{correct,wrong}/<idx>.png        the actual image tiles
  - figures/confusion_matrix.png            per-class confusion matrix
  - metrics.json                            overall val accuracy + per-class

Usage:
    python scripts/clip_zeroshot_qualitative.py \\
        --checkpoint runs/clip_eurosat_learned/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from basics.text_encoder import FrozenTextEncoder
from basics.vit import ViT
from vlm.clip import ProjectionHeads
from vlm.data import EUROSAT_CLASSES, build_eurosat_loaders


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("runs/clip_qualitative"))
    p.add_argument("--num-correct", type=int, default=5)
    p.add_argument("--num-wrong", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def denormalize(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img.cpu() * std + mean).clamp(0, 1)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "images" / "correct").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "images" / "wrong").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # ---------------------------------------------------------------- model
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    pos_encoding = ckpt.get("pos_encoding", "learned")
    vit = ViT(**cfg["vit"], pos_encoding=pos_encoding)
    vit.load_state_dict(ckpt["vit"])
    vit = vit.to(device).eval()

    text_encoder = FrozenTextEncoder(cfg["text_encoder"]["model_name"]).to(device)
    projection = ProjectionHeads(
        d_image=cfg["vit"]["d_model"],
        d_text=text_encoder.embedding_dim,
        d_proj=cfg["projection"]["d_proj"],
    ).to(device)
    projection.load_state_dict(ckpt["projection"])
    projection.eval()

    class_prompts = [f"a satellite image of {c}" for c in EUROSAT_CLASSES]
    with torch.no_grad():
        text_embeds = text_encoder(class_prompts)
        _, class_proj = projection(
            torch.zeros(len(class_prompts), cfg["vit"]["d_model"], device=device),
            text_embeds,
        )
        class_proj = F.normalize(class_proj, dim=-1)

    # ---------------------------------------------------------------- data
    _, val_dl, _ = build_eurosat_loaders(
        img_size=cfg["vit"]["img_size"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=0,
    )

    # ---------------------------------------------------------------- eval
    correct_examples: list[dict] = []
    wrong_examples: list[dict] = []
    overall_correct = 0
    overall_total = 0
    confusion = torch.zeros(len(EUROSAT_CLASSES), len(EUROSAT_CLASSES), dtype=torch.long)

    with torch.no_grad():
        for images, captions in val_dl:
            images = images.to(device, non_blocking=True)
            labels = torch.tensor(
                [class_prompts.index(c) for c in captions], device=device
            )
            feats = vit(images)
            img_proj, _ = projection(feats, torch.zeros_like(text_embeds[:1]))
            img_proj = F.normalize(img_proj, dim=-1)
            sims = img_proj @ class_proj.T  # (B, num_classes)
            top3 = sims.topk(3, dim=-1)
            preds = top3.indices[:, 0]

            for i in range(images.shape[0]):
                gt = int(labels[i].item())
                pred = int(preds[i].item())
                confusion[gt, pred] += 1
                row = {
                    "image_idx": overall_total + i,
                    "gold": EUROSAT_CLASSES[gt],
                    "top3": [EUROSAT_CLASSES[c] for c in top3.indices[i].tolist()],
                    "top3_sim": [float(s) for s in top3.values[i].tolist()],
                    "image": images[i].detach().cpu(),
                }
                if pred == gt and len(correct_examples) < args.num_correct:
                    correct_examples.append(row)
                elif pred != gt and len(wrong_examples) < args.num_wrong:
                    wrong_examples.append(row)
            overall_correct += (preds == labels).sum().item()
            overall_total += labels.size(0)
            if len(correct_examples) >= args.num_correct and len(wrong_examples) >= args.num_wrong and overall_total >= 2000:
                break

    accuracy = overall_correct / max(overall_total, 1)

    # ---------------------------------------------------------------- save
    from torchvision.utils import save_image

    rows: list[dict] = []
    for ex in correct_examples + wrong_examples:
        is_correct = ex["gold"] == ex["top3"][0]
        bucket = "correct" if is_correct else "wrong"
        path = args.output_dir / "images" / bucket / f"{ex['image_idx']:05d}.png"
        save_image(denormalize(ex["image"]), path)
        rows.append({
            "image_path": str(path),
            "gold": ex["gold"],
            "top3": ex["top3"],
            "top3_similarity": ex["top3_sim"],
            "correct": is_correct,
        })
    with open(args.output_dir / "examples.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Confusion matrix figure
    confusion_norm = confusion.float() / confusion.sum(dim=-1, keepdim=True).clamp(min=1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(confusion_norm.numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(EUROSAT_CLASSES)))
    ax.set_yticks(range(len(EUROSAT_CLASSES)))
    ax.set_xticklabels(EUROSAT_CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(EUROSAT_CLASSES, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title(f"Zero-shot confusion (val acc {accuracy:.3f})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(args.output_dir / "figures" / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    per_class_acc = {
        EUROSAT_CLASSES[i]: float(confusion[i, i] / confusion[i].sum().clamp(min=1))
        for i in range(len(EUROSAT_CLASSES))
    }

    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump({
            "val_accuracy": accuracy,
            "per_class_accuracy": per_class_acc,
            "num_eval": overall_total,
            "checkpoint": str(args.checkpoint),
        }, f, indent=2)
    print(f"DONE  acc={accuracy:.4f}  -> {args.output_dir}")


if __name__ == "__main__":
    main()
