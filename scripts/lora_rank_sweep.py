"""§4.2 — LoRA rank sweep on RESISC45.

Trains LoRA adapters at ranks {1, 2, 4, 8, 16, 32, 64} (alpha = 2*r so the
scaling alpha/r is held constant) and plots test accuracy vs rank.

Reuses scripts/finetune_resisc.py end-to-end as a subprocess so each rank
gets its own runs/<dir>. Aggregates the metrics.json files into a single
plot at runs/lora_rank_sweep/.

Usage:
    python scripts/lora_rank_sweep.py \\
        --config configs/lora_resisc.yaml \\
        --pretrained runs/clip_eurosat_learned/best.pt \\
        --ranks 1 2 4 8 16 32 64
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--pretrained", type=Path, required=True)
    p.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--output-dir", type=Path, default=Path("runs/lora_rank_sweep"))
    p.add_argument("--device", default=None)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip ranks whose runs/<dir>/metrics.json already exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent
    finetune_script = repo_root / "scripts" / "finetune_resisc.py"

    rank_to_metrics: dict[int, dict] = {}
    for rank in args.ranks:
        alpha = 2 * rank
        run_dir = repo_root / "runs" / f"resisc_lora_rank{rank}_a{int(alpha)}"
        metrics_path = run_dir / "metrics.json"
        if args.skip_existing and metrics_path.exists():
            print(f"=== rank {rank} already done; skipping ===")
            rank_to_metrics[rank] = json.loads(metrics_path.read_text())
            continue
        print(f"=== rank {rank}, alpha {alpha} ===")
        cmd = [
            sys.executable, str(finetune_script),
            "--config", str(args.config),
            "--method", "lora",
            "--rank", str(rank),
            "--alpha", str(alpha),
            "--pretrained", str(args.pretrained),
            "--output-dir", str(run_dir),
        ]
        if args.device:
            cmd += ["--device", args.device]
        subprocess.run(cmd, check=True)
        rank_to_metrics[rank] = json.loads(metrics_path.read_text())

    # ---------------------------------------------------------------- plot
    ranks_sorted = sorted(rank_to_metrics.keys())
    accs = [rank_to_metrics[r]["best_test_acc"] for r in ranks_sorted]
    params = [rank_to_metrics[r]["trainable_params"] for r in ranks_sorted]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ranks_sorted, accs, "o-", color="steelblue", label="test accuracy")
    ax1.set_xlabel("LoRA rank r (alpha = 2r)")
    ax1.set_ylabel("test accuracy", color="steelblue")
    ax1.set_xscale("log", base=2)
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(ranks_sorted, params, "s--", color="tomato", label="trainable params")
    ax2.set_ylabel("trainable params", color="tomato")
    ax2.set_yscale("log")
    fig.suptitle("RESISC45 — LoRA rank sweep")
    fig.tight_layout()
    fig.savefig(args.output_dir / "figures" / "rank_sweep.png", dpi=120)
    plt.close(fig)

    summary = {
        "ranks": ranks_sorted,
        "test_accuracy": accs,
        "trainable_params": params,
        "by_rank": {str(r): rank_to_metrics[r] for r in ranks_sorted},
    }
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"DONE -> {args.output_dir}")


if __name__ == "__main__":
    main()
