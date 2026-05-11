"""§5 — Qualitative evaluation of a trained VLM.

Saves under <output_dir>/:
  - examples.jsonl                    one row per qualitative sample
  - metrics.json                      overall + per-q_type accuracy
  - figures/acc_by_qtype.png          per-q_type accuracy bar chart
  - images/<idx>.png                  saved images (with --save-images)

Usage:
    python scripts/eval_vlm.py \\
        --checkpoint runs/vlm_all_patches_image_bidir_A/best.pt \\
        --num-examples 10 --save-images
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._plot_utils import bar_plot
from vlm.data import build_clevr_loaders
from vlm.eval import batch_clevr_accuracy, clevr_exact_match
from vlm.model import VisionLanguageModel
from vlm.projector import VisionLanguageProjector
from basics.vit import ViT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--num-examples", type=int, default=10)
    p.add_argument("--max-eval", type=int, default=500)
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path("runs/vlm_qualitative"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    if args.save_images:
        (args.output_dir / "images").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # ---------------------------------------------------------------- ckpt
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    saved_args = ckpt["args"]
    injection = saved_args["injection"]

    # ---------------------------------------------------------------- model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_str = cfg["decoder"]["torch_dtype"]
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        dtype_str
    ]

    attn_impl = cfg["decoder"].get("attn_implementation", "sdpa")
    try:
        decoder = AutoModelForCausalLM.from_pretrained(
            cfg["decoder"]["model_name"],
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
        )
    except (ImportError, ValueError) as e:
        if "flash" in str(e).lower() and attn_impl != "sdpa":
            print(f"[warn] {attn_impl} unavailable ({e}); falling back to sdpa")
            decoder = AutoModelForCausalLM.from_pretrained(
                cfg["decoder"]["model_name"],
                torch_dtype=torch_dtype,
                attn_implementation="sdpa",
            )
        else:
            raise
    tokenizer = AutoTokenizer.from_pretrained(cfg["decoder"]["model_name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    image_token_id = ckpt.get("image_token_id")
    if image_token_id is not None:
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        decoder.resize_token_embeddings(len(tokenizer))

    # We don't have the §3 ckpt's vit_cfg here, so we recover it from the saved
    # ViT state dict shapes via the original config baked into best.pt. The
    # train script saves cfg (vlm config) but not vit cfg directly; SmolLM
    # decoder dim is fixed, so we reconstruct ViT from any image-encoder ckpt
    # if available, else trust the saved state_dict shapes.
    # We rely on the train script's saved vit state_dict.
    vit_state = ckpt["vit"]
    # Infer vit hyperparams from the state dict — fall back to a reasonable default.
    d_model = vit_state["cls_token"].shape[-1]
    num_patches_plus1 = vit_state["pos_embed"].shape[1]
    num_patches = num_patches_plus1 - 1
    # Patch size: pos_embed has shape (1, num_patches+1, d_model); we don't know img/patch.
    # Use config-default img_size=64, infer patch_size from num_patches.
    grid = int(num_patches**0.5)
    img_size = 64
    patch_size = img_size // grid
    num_blocks = sum(1 for k in vit_state.keys() if k.startswith("blocks.") and k.endswith(".ln1.weight"))
    num_heads = (
        sum(1 for k in vit_state.keys() if k.startswith("blocks.0.attn.heads.") and k.endswith(".q_proj.weight"))
    )
    vit = ViT(img_size=img_size, patch_size=patch_size, d_model=d_model,
              num_heads=num_heads, num_blocks=num_blocks, dropout=0.0)
    vit.load_state_dict(vit_state)

    d_decoder = decoder.get_input_embeddings().weight.shape[1]
    projector = VisionLanguageProjector(
        d_image=d_model, d_decoder=d_decoder, expansion=cfg["projector"]["expansion"],
    )
    projector.load_state_dict(ckpt["projector"])
    decoder.load_state_dict(ckpt["decoder"])

    vit = vit.to(device).to(torch_dtype)
    projector = projector.to(device).to(torch_dtype)
    decoder = decoder.to(device)

    model = VisionLanguageModel(vit, projector, decoder, tokenizer, image_token_id=image_token_id)
    model.eval()

    # ---------------------------------------------------------------- data
    _, val_dl = build_clevr_loaders(img_size=img_size, batch_size=8, num_workers=0)

    # ---------------------------------------------------------------- eval loop
    preds, golds, q_types = [], [], []
    raw_examples: list[dict] = []
    seen = 0
    with torch.no_grad():
        for batch in val_dl:
            images = batch["image"].to(device).to(torch_dtype)
            questions = batch["question"]
            answers = batch["answer"]
            qts = batch["q_type"]
            if injection == "interleaved":
                prompts = [f"<image> Question: {q} Answer:" for q in questions]
            else:
                prompts = [f"Question: {q} Answer:" for q in questions]
            outs = model.generate(images, prompts, injection=injection, **cfg["generation"])
            if seen == 0:
                print("---- first batch of predictions ----")
                for q, a, p in zip(questions[:5], answers[:5], outs[:5]):
                    print(f"  Q: {q!r}")
                    print(f"  A: {a!r}")
                    print(f"  pred: {p!r}")
                    print()
            preds.extend(outs)
            golds.extend(answers)
            q_types.extend(qts)
            for i, (q, a, p, qt) in enumerate(zip(questions, answers, outs, qts)):
                raw_examples.append({
                    "question": q, "gold": a, "prediction": p,
                    "q_type": qt,
                    "correct": clevr_exact_match(p, a),
                    "image": images[i].detach().cpu(),
                })
            seen += len(questions)
            if seen >= args.max_eval:
                break

    accuracy = batch_clevr_accuracy(preds, golds, q_types)
    print("Accuracy summary:", json.dumps(accuracy, indent=2))

    # ---------------------------------------------------------------- dump
    qual_idxs = list(range(min(args.num_examples, len(raw_examples))))
    with open(args.output_dir / "examples.jsonl", "w") as f:
        for i in qual_idxs:
            ex = raw_examples[i]
            row = {
                "idx": i,
                "question": ex["question"],
                "gold": ex["gold"],
                "prediction": ex["prediction"],
                "q_type": ex["q_type"],
                "correct": ex["correct"],
            }
            if args.save_images:
                from torchvision.utils import save_image

                # Undo ImageNet normalization for visualization
                img = ex["image"].to(torch.float32)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = (img * std + mean).clamp(0, 1)
                img_path = args.output_dir / "images" / f"{i:04d}.png"
                save_image(img, img_path)
                row["image_path"] = str(img_path)
            f.write(json.dumps(row) + "\n")

    per_qtype = {k: v for k, v in accuracy.items() if k != "overall"}
    if per_qtype:
        bar_plot(
            list(per_qtype.keys()),
            list(per_qtype.values()),
            args.output_dir / "figures" / "acc_by_qtype.png",
            title=f"Per-q_type accuracy (overall {accuracy['overall']:.3f})",
            ylabel="accuracy",
        )

    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump({
            "accuracy": accuracy,
            "num_eval": seen,
            "checkpoint": str(args.checkpoint),
            "injection": injection,
        }, f, indent=2)

    print(f"DONE  overall={accuracy['overall']:.4f}  -> {args.output_dir}")


if __name__ == "__main__":
    main()
