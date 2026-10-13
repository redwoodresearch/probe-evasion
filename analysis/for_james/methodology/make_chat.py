"""Everyday chat against the maths and ordering probes: the dot plot with an everyday-chat row and both flag lines, plus
example chat replies at chosen score quantiles.

    .venv/bin/python analysis/for_james/methodology/make_chat.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from make_dots import COLORS, OUT, TASKS, flag_line, read, strips

BROAD = Path("missions/m50_realistic_monitor/runs/broad_scores.jsonl")  # 1,720 ordinary Qwen chat replies (m27's pool)
BROAD_LINES = Path("missions/m50_realistic_monitor/lines.json")
BROAD_LINE_KEY = "broad_cal_iv_1pct"  # m50's primary broad line
CHAT_COLOR = "#8a6d3b"
EXAMPLE_QUANTILES = (0.5, 0.9, 1.0)
EXAMPLE_CHARS = 400


def chat_rows() -> list[dict]:
    return [r for r in read(BROAD) if r["context"] == "none"]  # no instruction in view


def plot() -> None:
    rows = chat_rows()
    broad_lines = json.loads(BROAD_LINES.read_text())
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=True)
    for ax, task in zip(axes, ("maths", "ordering")):
        cfg = TASKS[task]
        strict = flag_line(cfg)
        broad = broad_lines[task][BROAD_LINE_KEY]["line"]
        chat = np.array([r["peaks"][cfg.probe] for r in rows])
        data = [("everyday chat\n(no task)", chat, CHAT_COLOR)] + strips(cfg)
        for i, (label, xs, color) in enumerate(data):
            y = len(data) - 1 - i
            ax.scatter(xs, y + rng.uniform(-0.28, 0.28, len(xs)), s=4, color=color, alpha=0.25, linewidths=0)
            med = float(np.median(xs))
            ax.plot([med, med], [y - 0.38, y + 0.38], color=color, lw=3)
            ax.text(med, y + 0.42, f"{med:+.1f}", ha="center", va="bottom", fontsize=8, color=color)
        for x, name, style, side in ((strict, "strict 1% line (look-alikes) ", "-", "right"),
                                     (broad, " broad 1% line (everyday chat)", "--", "left")):
            ax.axvline(x, color=COLORS["line"], lw=1.4, ls=style)
            ax.text(x, -0.62, name, ha=side, va="bottom", fontsize=7, color=COLORS["line"])
        share = float(np.mean(chat > strict))
        ax.set_title(f"{cfg.title}: the strict line flags {share:.0%} of everyday chat", loc="left", fontsize=10, fontweight="bold")
        ax.set_yticks(range(len(data)), [d[0] for d in reversed(data)], fontsize=9)
        ax.set_ylim(-0.75, len(data) - 0.1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[-1].set_xlabel("probe score (log-odds); higher = 'doing the task'", fontsize=9)
    fig.suptitle("Everyday chat: 1,720 ordinary Qwen3.5-27B replies, no instruction. Other rows as in each probe's dot plot "
                 "(Fable 5.1, frozen test, B = 16)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "chat_dots.png", dpi=160)


def examples() -> str:
    rows = chat_rows()
    out = ["# Everyday chat examples (m50 broad pool, no instruction in view)\n"]
    for task in ("maths", "ordering"):
        cfg = TASKS[task]
        strict = flag_line(cfg)
        ranked = sorted(rows, key=lambda r: r["peaks"][cfg.probe])
        out.append(f"\n## {cfg.title} probe (strict line {strict:+.2f})\n")
        for q in EXAMPLE_QUANTILES:
            r = ranked[min(int(q * len(ranked)), len(ranked) - 1)]
            reply = r["reply"].replace("\n", " ")
            cut = reply[:EXAMPLE_CHARS] + (" …" if len(reply) > EXAMPLE_CHARS else "")
            out.append(f"- **{'top' if q == 1.0 else f'{q:.0%} quantile'}** ({r['topic']}), score {r['peaks'][cfg.probe]:+.2f}\n"
                       f"  - request: {r['request']}\n  - reply: {cut}\n")
    return "\n".join(out)


def main() -> None:
    plot()
    (OUT.parent / "chat_examples.md").write_text(examples())


if __name__ == "__main__":
    main()
