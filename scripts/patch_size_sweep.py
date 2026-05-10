"""§2.4 — Patch-size sweep with forward-pass timing.

Builds a ViT with d_model=384, num_heads=6, num_blocks=6 for each patch size
P ∈ {8, 16, 32} on 224×224 inputs, and times the forward pass on a batch of 16.
Saves under <output_dir>/:
  - metrics.json   {patch_size: {num_patches, mean_ms, std_ms}}
  - figures/patch_size_time.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from basics.vit import ViT
from scripts._plot_utils import bar_plot


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("runs/patch_size_sweep"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--patch-sizes", type=int, nargs="+", default=[8, 16, 32])
    return p.parse_args()


def time_forward(model: torch.nn.Module, x: torch.Tensor, warmup: int, steps: int) -> tuple[float, float]:
    device = x.device
    is_cuda = device.type == "cuda"
    for _ in range(warmup):
        _ = model(x)
    if is_cuda:
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(steps):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(x)
        if is_cuda:
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000)
    times = torch.tensor(times_ms)
    return float(times.mean()), float(times.std())


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    results: dict[str, dict] = {}

    for P in args.patch_sizes:
        if args.img_size % P != 0:
            print(f"skip P={P}: img_size {args.img_size} not divisible")
            continue
        N = (args.img_size // P) ** 2
        print(f"=== patch_size={P}  num_patches={N} ===")

        model = ViT(
            img_size=args.img_size, patch_size=P, d_model=384,
            num_heads=6, num_blocks=6, dropout=0.0,
        ).to(device).eval()
        x = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)

        with torch.no_grad():
            mean_ms, std_ms = time_forward(model, x, args.warmup, args.steps)
        print(f"  forward: {mean_ms:.2f} ± {std_ms:.2f} ms")
        results[str(P)] = {"num_patches": N, "mean_ms": mean_ms, "std_ms": std_ms}

        del model, x
        if device.type == "cuda":
            torch.cuda.empty_cache()

    bar_plot(
        [f"P={p}\nN={results[p]['num_patches']}" for p in results],
        [results[p]["mean_ms"] for p in results],
        args.output_dir / "figures" / "patch_size_time.png",
        title=f"ViT forward time at {args.img_size}x{args.img_size} (B={args.batch_size})",
        ylabel="ms / forward pass",
    )

    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump({
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "warmup": args.warmup,
            "steps": args.steps,
            "device": str(device),
            "results": results,
        }, f, indent=2)
    print(f"DONE -> {args.output_dir}")


if __name__ == "__main__":
    main()
