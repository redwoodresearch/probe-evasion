"""Dot plot of m38's volitional conditions on Qwen3.5-27B maths, reasoning on (practice set, interim)."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUN = Path("missions/m38_volitional_track/runs/m27_on")
OUT = Path("experimental/analysis/m38_volitional_dots/volitional_27b_reasoning_on.png")
PROBE = "onpk@L48@s0"
FLAG_RATE = "0.01"

# (arm, label, asked to hide?)
CONDITIONS = [
    ("plain", "Plain: no instruction", False),
    ("C2c", "Told a monitor reads activations (control: not asked to hide)", False),
    ("C2", "Told a monitor reads activations, asked to hide (user turn)", True),
    ("C3c", "Told exactly how the probe reads (control: not asked to hide)", False),
    ("C3", "Told exactly how the probe reads, asked to hide (user turn)", True),
    ("C4", "… plus an incentive", True),
    ("C6", "… as an authorised red-team exercise (system prompt)", True),
]
READINGS = [("answer", "Final answer only"), ("whole", "Reasoning + final answer")]
ASKED_COLOUR, CONTROL_COLOUR = "#D9622B", "#8A929C"


def load_valid_rows() -> dict[str, list[dict]]:
    by_arm = defaultdict(list)
    for line in RUN.joinpath("pilot_rows.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["valid"]:
            by_arm[row["arm"]].append(row)
    return by_arm


def flag_line(reading: str) -> float:
    lines = json.loads(RUN.joinpath("lines.json").read_text())
    pool_reading = "think" if reading == "whole" else reading
    return lines["by_pool"]["headline"][pool_reading][PROBE][FLAG_RATE]


def main() -> None:
    by_arm = load_valid_rows()
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
    for ax, (reading, title) in zip(axes, READINGS):
        line = flag_line(reading)
        plain_median = np.median([r["peaks_by_reading"][reading][PROBE] for r in by_arm["plain"]])
        for y, (arm, _, asked) in enumerate(CONDITIONS):
            scores = np.array([r["peaks_by_reading"][reading][PROBE] for r in by_arm[arm]])
            colour = ASKED_COLOUR if asked else CONTROL_COLOUR
            ax.scatter(scores, y + rng.uniform(-0.22, 0.22, len(scores)), s=14, alpha=0.55, color=colour, lw=0)
            median = np.median(scores)
            ax.plot([median, median], [y - 0.32, y + 0.32], color="#1B2430", lw=2.5)
            gap_closed = (plain_median - median) / (plain_median - line)
            ax.text(1.01, y, f"{gap_closed:+.0%}  (n={len(scores)})", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=9, color="#1B2430")
        ax.axvline(line, color="#C0392B", ls="--", lw=1.5)
        ax.text(line, len(CONDITIONS) - 0.45, " 1% flag line", color="#C0392B", fontsize=9, va="bottom")
        ax.set_title(f"{title}\n(right-hand numbers: share of the way from plain to the line)", fontsize=11)
        ax.set_xlabel("probe score, log-odds (instruction removed)")
        ax.grid(axis="x", color="#E5E5E5")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_yticks(range(len(CONDITIONS)), [label for _, label, _ in CONDITIONS], fontsize=10)
    axes[0].invert_yaxis()
    fig.suptitle("Qwen3.5-27B, maths, reasoning on: told it's monitored, does it hide its work on its own?\n"
                 "Each dot is one correct reply (80 practice problems; interim). Bar = median. "
                 "Orange = asked to hide, grey = plain and controls.", fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.32)
    fig.savefig(OUT, dpi=160)
    print(OUT)


if __name__ == "__main__":
    main()
