"""Shared helpers for logging metrics and saving plots inside a run dir.

Every script writes to <output_dir>/{metrics.json, figures/*.png, log.csv}.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, safe in Colab/SSH
import matplotlib.pyplot as plt


class RunLogger:
    """Append-only step/epoch logger that also writes a CSV row per call."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []
        self._csv_path = self.output_dir / "log.csv"
        self._csv_keys: list[str] | None = None

    def log(self, **kwargs) -> None:
        self.records.append(kwargs)
        if self._csv_keys is None:
            self._csv_keys = list(kwargs.keys())
            with open(self._csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self._csv_keys).writeheader()
        with open(self._csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._csv_keys).writerow(
                {k: kwargs.get(k, "") for k in self._csv_keys}
            )

    def series(self, key: str) -> list:
        return [r[key] for r in self.records if key in r]

    def save_metrics(self, metrics: dict) -> None:
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    def line_plot(
        self,
        x_key: str,
        y_keys: list[str] | str,
        filename: str,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        logy: bool = False,
    ) -> Path:
        if isinstance(y_keys, str):
            y_keys = [y_keys]
        fig, ax = plt.subplots(figsize=(7, 4))
        for k in y_keys:
            xs, ys = [], []
            for r in self.records:
                if x_key in r and k in r and r[k] is not None:
                    xs.append(r[x_key])
                    ys.append(r[k])
            if xs:
                ax.plot(xs, ys, label=k, marker="o" if len(xs) < 50 else None, linewidth=1.5)
        ax.set_xlabel(xlabel or x_key)
        ax.set_ylabel(ylabel or (y_keys[0] if len(y_keys) == 1 else "value"))
        if title:
            ax.set_title(title)
        if logy:
            ax.set_yscale("log")
        if len(y_keys) > 1:
            ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = self.figures_dir / filename
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return out


def bar_plot(
    labels: list[str],
    values: list[float],
    filename: Path,
    title: str | None = None,
    ylabel: str | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color="steelblue")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.3g}",
                ha="center", va="bottom", fontsize=9)
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    return filename


def cosine_with_warmup(step: int, total_steps: int, warmup_steps: int) -> float:
    """LR multiplier in [0, 1]."""
    import math

    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
