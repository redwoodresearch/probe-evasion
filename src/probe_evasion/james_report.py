"""One chart for the update to James: per probe, the share of ALL attempts that both do the task and get past the probe,
for a Claude-optimized prompt vs the plain prompt (frozen prompts, held-out test tasks, EMA-peak reading, 1% false-alarm
line from cleaned no-concept replies). Only probes trained for the reading they are deployed with (BCE on the EMA peak).

    uv run python -m probe_evasion.james_report   # writes share/james_evasion_bars.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from probe_evasion.proposal_figures import GREY, INK, ORANGE

SARCASM, ANGER = Path("experimental/missions/m3a_sarcasm"), Path("experimental/missions/m3c_anger")
FALSE_ALARM = 0.01
LEFT, RIGHT = 0.36, 0.98
HEADER_X = (0.012 - LEFT) / (RIGHT - LEFT)  # group headings start at the figure's left edge (axes units)

Arm = tuple[int, int, int]  # (valid and missed, valid, attempts)


def jl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open()]


def sarcasm(layer: int, removed: bool = False) -> tuple[Arm, Arm]:
    """Counted per attempt (repeated identical replies included), unlike RESULT.md's distinct-reply counts."""
    probe = f"offpk@L{layer}"
    calibration = [x["scores"][probe]["ema"] for x in jl(SARCASM / "scores_calibration.jsonl") if x["row"]["clean"]]
    line = np.quantile(calibration, 1 - FALSE_ALARM)
    rows = jl(SARCASM / "phase_c.jsonl")
    key = "scores_removed" if removed else "scores"

    def arm(name: str) -> Arm:
        rs = [r for r in rows if r["arm"] == name]
        valid = [r for r in rs if r["present"]]
        return sum(r[key][probe]["ema"] <= line for r in valid), len(valid), len(rs)

    return arm("searched"), arm("plain")


def anger(layer: int, removed: bool = False) -> tuple[Arm, Arm]:
    probe = json.loads((ANGER / "v2_frozen_search_summary.json").read_text())[f"claude_peak:anger@L{layer}"]
    key = f"evaded_no_instr@{FALSE_ALARM}" if removed else f"evaded@{FALSE_ALARM}"
    return tuple((probe[a][key], probe[a]["valid"], probe[a]["n"]) for a in ("optimized (frozen search #4)", "plain"))


# (group, label, data)
ROWS = [
    ("Sarcasm: probe trained on Claude-written replies, Qwen3.5-27B", [
        ("Layer 48: attack found against a sister probe", lambda: sarcasm(48)),
        ("    same answers, instruction removed", lambda: sarcasm(48, removed=True)),
        ("Layer 16: attack found against a sister probe", lambda: sarcasm(16)),
        ("    same answers, instruction removed", lambda: sarcasm(16, removed=True)),
    ]),
    ("Anger: probe trained on Claude-written replies, Qwen3.5-27B", [
        ("Layer 48: attacked directly", lambda: anger(48)),
        ("    same answers, instruction removed", lambda: anger(48, removed=True)),
        ("Layer 16: the layer-48 attack, re-scored", lambda: anger(16)),
    ]),
]


def main(out: Path = Path("experimental/share/james_evasion_bars.png")) -> Path:
    items = []
    for group, rows in ROWS:
        items.append(("group", group, None))
        items += [("row", label, get()) for label, get in rows]
    fig, ax = plt.subplots(figsize=(11.5, 0.5 * len(items) + 2.4), dpi=170)
    y = 0
    for kind, label, data in items:
        if kind == "group":
            y += 0.3
            ax.text(HEADER_X, -y, label, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=9.5,
                    fontweight="bold", color=INK)
            y += 0.8
            continue
        (k, v, n), (pk, pv, pn) = data
        rate, base = k / n, pk / pn
        ax.barh(-y, rate, height=0.62, color=ORANGE)
        ax.barh(-y, base, height=0.3, color=GREY)
        ax.text(-0.015, -y, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9, color=INK)
        ax.text(max(rate, base) + 0.012, -y, f"{100 * rate:.0f}%  ({k}/{n}; task done {v}/{n})\nplain: {pk}/{pn} "
                f"(task done {pv}/{pn})", va="center", fontsize=8.3, color=INK, linespacing=1.3)
        y += 1
    ax.set_xlim(0, 1.42)
    ax.set_xticks([0, .2, .4, .6, .8, 1])
    ax.set_ylim(-y + 0.3, 0.2)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.set_xlabel("Attempts that do the task AND get past the probe (1% false-alarm line, held-out test tasks)", fontsize=9.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.legend(handles=[Patch(color=ORANGE, label="Optimized prompt (found by a Claude search)"),
                        Patch(color=GREY, label="Plain prompt")], loc="upper left", bbox_to_anchor=(0.006, 0.87), ncol=2,
               fontsize=9, frameon=False)
    fig.text(0.012, 0.965, "Prompted probe evasion against probes trained for the reading they are deployed with",
             fontsize=12, fontweight="bold", color=INK)
    fig.text(0.012, 0.915, "Probes trained on the EMA-peak score they are read with. Attacked directly: Claude optimized the prompt "
             "against this probe's scores. Sister probe: same\ntraining data, trained for the average reading. Instruction removed: "
             "the same answers scored without the prompt's instruction in the probe's input.", fontsize=8.6, color=GREY,
             linespacing=1.4)
    fig.text(0.012, 0.015, "Sarcasm prompt never mentions a probe (one flat sentence that flips reality); the anger prompt does, but the "
             "same style without that mention does as well (197/200).\nAnger: the evading replies are mild. A stricter judge "
             "('clearly more than mildly frustrated') passes 32 of the 196; all 32 evade in context, 15 with the instruction removed.\n"
             "Not shown: maths and chemistry probes from earlier, which were trained on the average and read with the peak.",
             fontsize=8.2, color=GREY, linespacing=1.45)
    fig.subplots_adjust(left=LEFT, right=RIGHT, top=0.8, bottom=0.2)
    fig.savefig(out)
    return out


if __name__ == "__main__":
    print(main())
