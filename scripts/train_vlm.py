"""§5 — VLM training on CLEVR.

Saves under <output_dir>/:
  - log.csv             per-step / per-eval metrics
  - metrics.json        final summary
  - figures/loss.png    train loss vs step
  - figures/lr.png      learning rate vs step
  - figures/grad_norm.png
  - figures/val_acc.png val exact-match accuracy vs step
  - figures/val_acc_by_qtype.png  per-q_type bar chart at best step
  - best.pt             best projector + decoder + (optional) vit state dicts

Usage:
    python scripts/train_vlm.py --config configs/vlm_clevr.yaml \\
        --pretrained-vit runs/clip_eurosat/best.pt \\
        --injection all_patches --mask-mode image_bidir --freeze-config A
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from basics.lora import apply_lora_to_attention
from basics.vit import ViT
from scripts._plot_utils import RunLogger, bar_plot, cosine_with_warmup
from vlm.data import build_clevr_loaders
from vlm.eval import batch_clevr_accuracy
from vlm.model import VisionLanguageModel
from vlm.projector import VisionLanguageProjector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--pretrained-vit", type=Path, required=True)
    p.add_argument("--injection", choices=["cls", "all_patches", "interleaved"], default="all_patches")
    p.add_argument("--mask-mode", choices=["causal", "image_bidir"], default="causal")
    p.add_argument(
        "--freeze-config",
        choices=["A", "B", "C", "D"],
        default="A",
        help="A=projector only; B=projector+decoder LoRA; C=projector+full decoder; D=everything",
    )
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def apply_freeze_config(
    config: str, vit, projector, decoder, lora_rank: int, lora_alpha: float
) -> None:
    """Set requires_grad according to the chosen freeze config."""
    # Default: freeze everything, then unfreeze.
    for p in vit.parameters():
        p.requires_grad_(False)
    for p in projector.parameters():
        p.requires_grad_(False)
    for p in decoder.parameters():
        p.requires_grad_(False)

    # Projector is always trained.
    for p in projector.parameters():
        p.requires_grad_(True)

    if config == "A":
        pass  # projector only
    elif config == "B":
        # Apply LoRA to decoder attention if it has q/v projections we can target.
        # SmolLM2 uses standard transformer naming; we use peft if available.
        try:
            from peft import LoraConfig, get_peft_model

            lora_cfg = LoraConfig(
                r=lora_rank, lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            )
            decoder = get_peft_model(decoder, lora_cfg)  # noqa: F841
            print("[freeze_config B] applied PEFT LoRA to decoder q/v")
        except ImportError:
            print("[freeze_config B] peft not installed, falling back to full FT")
            for p in decoder.parameters():
                p.requires_grad_(True)
    elif config == "C":
        for p in decoder.parameters():
            p.requires_grad_(True)
    elif config == "D":
        for p in vit.parameters():
            p.requires_grad_(True)
        for p in decoder.parameters():
            p.requires_grad_(True)


def evaluate(model, val_loader, injection, max_examples, device, generation_kwargs):
    model.eval()
    preds, golds, q_types = [], [], []
    seen = 0
    image_dtype = next(model.vit.patch_embed.proj.parameters()).dtype
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device).to(image_dtype)
            questions = batch["question"]
            answers = batch["answer"]
            # Plain prompt format; for "interleaved" mode prepend the <image> token.
            if injection == "interleaved":
                prompts = [f"<image> Question: {q} Answer:" for q in questions]
            else:
                prompts = [f"Question: {q} Answer:" for q in questions]
            outs = model.generate(images, prompts, injection=injection, **generation_kwargs)
            if seen == 0:
                print(f"  sample preds: {list(zip(answers[:3], outs[:3]))}")
            preds.extend(outs)
            golds.extend(answers)
            q_types.extend(batch["q_type"])
            seen += len(questions)
            if seen >= max_examples:
                break
    return batch_clevr_accuracy(preds, golds, q_types)


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.output_dir is None:
        args.output_dir = Path("runs") / f"vlm_{args.injection}_{args.mask_mode}_{args.freeze_config}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(args.output_dir)
    device = torch.device(args.device)

    # ---------------------------------------------------------------- data
    train_dl, val_dl = build_clevr_loaders(
        img_size=64,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )

    # ---------------------------------------------------------------- ViT
    ckpt = torch.load(args.pretrained_vit, map_location="cpu")
    vit_cfg = ckpt["config"]["vit"]
    pos_encoding = ckpt.get("pos_encoding", "learned")
    vit = ViT(**vit_cfg, pos_encoding=pos_encoding)
    vit.load_state_dict(ckpt["vit"])
    vit = vit.to(device)

    # ---------------------------------------------------------------- decoder
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
        ).to(device)
    except (ImportError, ValueError) as e:
        if "flash" in str(e).lower() and attn_impl != "sdpa":
            print(f"[warn] {attn_impl} unavailable ({e}); falling back to sdpa")
            decoder = AutoModelForCausalLM.from_pretrained(
                cfg["decoder"]["model_name"],
                torch_dtype=torch_dtype,
                attn_implementation="sdpa",
            ).to(device)
        else:
            raise
    tokenizer = AutoTokenizer.from_pretrained(cfg["decoder"]["model_name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    image_token_id = None
    if args.injection == "interleaved":
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        decoder.resize_token_embeddings(len(tokenizer))
        image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    # ---------------------------------------------------------------- projector
    d_decoder = decoder.get_input_embeddings().weight.shape[1]
    projector = VisionLanguageProjector(
        d_image=vit_cfg["d_model"],
        d_decoder=d_decoder,
        expansion=cfg["projector"]["expansion"],
    ).to(device).to(torch_dtype)
    vit = vit.to(torch_dtype)

    apply_freeze_config(args.freeze_config, vit, projector, decoder,
                        args.lora_rank, args.lora_alpha)

    model = VisionLanguageModel(vit, projector, decoder, tokenizer, image_token_id=image_token_id)

    # ---------------------------------------------------------------- optim
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
        betas=tuple(cfg["optim"]["betas"]),
    )
    base_lr = cfg["optim"]["lr"]
    warmup_steps = cfg["optim"]["warmup_steps"]
    num_steps = cfg["train"]["num_steps"]
    accum = cfg["train"]["gradient_accumulation_steps"]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ---------------------------------------------------------------- train
    train_iter = iter(train_dl)
    step = 0
    best_acc = -1.0
    best_per_qtype: dict[str, float] = {}
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()

    while step < num_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            batch = next(train_iter)

        images = batch["image"].to(device).to(torch_dtype)
        questions = batch["question"]
        answers = batch["answer"]
        if args.injection == "interleaved":
            full_prompts = [f"<image> Question: {q} Answer: {a}{tokenizer.eos_token}"
                            for q, a in zip(questions, answers)]
            prefix_prompts = [f"<image> Question: {q} Answer:" for q in questions]
        else:
            full_prompts = [f"Question: {q} Answer: {a}{tokenizer.eos_token}"
                            for q, a in zip(questions, answers)]
            prefix_prompts = [f"Question: {q} Answer:" for q in questions]

        tok = tokenizer(full_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        prefix_tok = tokenizer(prefix_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        input_ids = tok["input_ids"]
        attention_mask = tok["attention_mask"]
        labels = input_ids.clone()
        # Mask padding from loss
        labels[attention_mask == 0] = -100
        # Answer-only loss: mask out question/prompt tokens per row so only the
        # answer (+ EOS) contributes to the loss. Without this, the model spends
        # most of its training signal predicting the question back to itself
        # and the actual answer tokens barely move the loss.
        prefix_lens = prefix_tok["attention_mask"].sum(dim=1)  # (B,)
        for b in range(input_ids.shape[0]):
            labels[b, : prefix_lens[b]] = -100

        out = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            injection=args.injection,
            mask_mode=args.mask_mode,
        )
        loss = out["loss"] / accum
        loss.backward()

        if (step + 1) % accum == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            lr_mult = cosine_with_warmup(step, num_steps, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr * lr_mult
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        else:
            grad_norm = torch.tensor(0.0)
            lr_mult = cosine_with_warmup(step, num_steps, warmup_steps)

        if step % cfg["train"]["log_every"] == 0:
            logger.log(
                step=step,
                loss=float(out["loss"].item()),
                lr=base_lr * lr_mult,
                grad_norm=float(grad_norm),
            )
            print(f"step {step} loss {out['loss'].item():.4f} lr {base_lr * lr_mult:.2e}")

        if step > 0 and step % cfg["train"]["eval_every_steps"] == 0:
            acc = evaluate(
                model, val_dl, args.injection,
                cfg["train"]["eval_max_examples"], device,
                cfg["generation"],
            )
            logger.log(step=step, val_acc=acc["overall"], **{f"acc_{k}": v for k, v in acc.items() if k != "overall"})
            print(f"[eval @ step {step}] {acc}")
            if acc["overall"] > best_acc:
                best_acc = acc["overall"]
                best_per_qtype = {k: v for k, v in acc.items() if k != "overall"}
                torch.save(
                    {
                        "projector": projector.state_dict(),
                        "vit": vit.state_dict(),
                        "decoder": decoder.state_dict(),
                        "config": cfg,
                        "args": vars(args),
                        "step": step,
                        "val_acc": acc,
                        "image_token_id": image_token_id,
                    },
                    args.output_dir / "best.pt",
                )
            model.train()

        step += 1

    elapsed = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0

    # ---------------------------------------------------------------- plots
    logger.line_plot("step", "loss", "loss.png",
                     title=f"VLM train loss ({args.injection}/{args.mask_mode}/{args.freeze_config})",
                     xlabel="step", ylabel="loss")
    logger.line_plot("step", "lr", "lr.png", title="LR schedule")
    logger.line_plot("step", "grad_norm", "grad_norm.png", title="Gradient norm")
    logger.line_plot("step", "val_acc", "val_acc.png",
                     title="Val exact-match accuracy", xlabel="step", ylabel="accuracy")

    if best_per_qtype:
        bar_plot(
            list(best_per_qtype.keys()),
            list(best_per_qtype.values()),
            args.output_dir / "figures" / "val_acc_by_qtype.png",
            title=f"Per-q_type accuracy at best step (overall {best_acc:.3f})",
            ylabel="accuracy",
        )

    metrics = {
        "best_val_acc": best_acc,
        "best_val_acc_by_qtype": best_per_qtype,
        "trainable_params": sum(p.numel() for p in trainable),
        "peak_memory_mb": peak_mem,
        "wall_time_sec": elapsed,
        "total_steps": step,
        "args": vars(args),
    }
    # Path objects are not JSON-serializable
    metrics["args"] = {
        k: (str(v) if isinstance(v, Path) else v) for k, v in metrics["args"].items()
    }
    logger.save_metrics(metrics)
    print(f"DONE  best_val_acc={best_acc:.4f}  time={elapsed:.0f}s  peak_mem={peak_mem:.0f}MB  "
          f"-> {args.output_dir}")


if __name__ == "__main__":
    main()
