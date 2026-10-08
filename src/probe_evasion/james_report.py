"""One chart for the update to James: per probe, the share of valid answers the probe misses when the model is told to
evade (Claude-optimized prompt, frozen, held-out problems) vs not told (plain prompt). EMA reading, 1% false-alarm line
from cleaned no-concept replies; a maths answer counts when the correct number appears however it is used.

    uv run python -m probe_evasion.james_report   # writes share/james_evasion_bars.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from matplotlib.patches import Patch

from probe_evasion.proposal_figures import GREY, INK, ORANGE

G, P = Path("results/gepa"), Path("results/concept_probes")
LEFT, RIGHT = 0.42, 0.98
HEADER_X = (0.012 - LEFT) / (RIGHT - LEFT)  # group headings start at the figure's left edge (axes units)


def read(p: Path) -> dict:
    return json.loads(p.read_text())


def frac(x: str) -> tuple[int, int]:
    return tuple(int(v) for v in x.split("/"))


def reval(run: str) -> tuple[tuple[int, int], tuple[int, int]]:
    m = read(G / run / "revalidated.json")["readings"]["ema"]["missed"]
    return ((m["optimized"]["missed_lenient"], m["optimized"]["valid_lenient"]),
            (m["no_instruction"]["missed_lenient"], m["no_instruction"]["valid_lenient"]))


def gemma_layer(layer: int) -> tuple[tuple[int, int], tuple[int, int]]:
    e = read(G / "gemma_maths_window_L36_20261007-154620" / "ladder_gemma_maths_layers.json")["results"][f"maths@L{layer}"]["ema"]
    return frac(e["missed_lenient"]["optimized"]), frac(e["missed_lenient"]["no_instruction"])


def chem_new() -> tuple[tuple[int, int], tuple[int, int]]:
    a = read(G / "qwen35_chemistry_ema_onpolicy_L48_20261008" / "summary.json")["arms"]
    return (a["optimized"]["missed_ema"], a["optimized"]["present"]), (a["no instruction"]["missed_ema"], a["no instruction"]["present"])


# (group, label, data); groups: how the probe's training data was made
ROWS = [
    ("Trained on the monitored model's own replies, labels checked against content", [
        ("Qwen maths, layer 48: attacked directly", lambda: reval("qwen35_maths_ema_onpolicyclean_L48_20261008-v2")),
        ("Qwen maths + varied division data, layer 48: attacked directly", lambda: reval("qwen35_maths_ema_onpolicydivaug_L48_20261008-v2")),
        ("Gemma maths, layer 36: attacked directly", lambda: reval("gemma_maths_ema_onpolicyclean_L36_20261008-v2")),
        ("Gemma maths, layer 30: attacked directly", lambda: reval("gemma_maths_ema_onpolicyclean_L30_20261008-v2")),
        ("Qwen chemistry (biology, maths as negatives), layer 48: attacked directly", chem_new),
    ]),
    ("Trained on text we wrote from templates (maths)", [
        ("Qwen maths, layer 48: attacked directly", lambda: reval("qwen35_maths_window_L48_20261007-173549")),
        ("Gemma maths, layer 36: attacked directly", lambda: gemma_layer(36)),
        ("Gemma maths, layer 12: layer-36 attack, re-scored", lambda: gemma_layer(12)),
        ("Gemma maths, layer 6 of 42: layer-36 attack, re-scored", lambda: gemma_layer(6)),
    ]),
]


def main(out: Path = Path("share/james_evasion_bars.png")) -> Path:
    items = []
    for group, rows in ROWS:
        items.append(("group", group, None))
        items += [("row", label, get()) for label, get in rows]
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(items) + 1.6), dpi=170)
    y = 0
    for kind, label, data in items:
        if kind == "group":
            y += 0.35
            ax.text(HEADER_X, -y, label, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=9.5,
                    fontweight="bold", color=INK)
            y += 0.75
            continue
        (k, n), (pk, pn) = data
        rate, base = k / n, pk / pn
        ax.barh(-y, rate, height=0.62, color=ORANGE)
        ax.barh(-y, base, height=0.3, color=GREY)
        ax.text(-0.015, -y, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9, color=INK)
        ax.text(max(rate, base) + 0.012, -y, f"{100 * rate:.0f}%  ({k}/{n})   not told: {pk}/{pn}", va="center", fontsize=8.5, color=INK)
        y += 1
    ax.set_xlim(0, 1.22)
    ax.set_xticks([0, .2, .4, .6, .8, 1])
    ax.set_ylim(-y + 0.3, 0.2)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.set_xlabel("Valid answers the probe misses (1% false-alarm line, held-out problems)", fontsize=9.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(handles=[Patch(color=ORANGE, label="Told to evade (Claude-optimized prompt)"),
                       Patch(color=GREY, label="Not told (plain prompt)")], loc="center right", fontsize=9, frameon=False)
    fig.text(0.012, 0.965, "Prompted probe evasion, by how the probe was trained", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.012, 0.935, "Attacked directly: Claude optimized the evasion prompt against this probe's own scores. Re-scored: the answers "
             "from the layer-36 attack, read by an earlier layer's probe.", fontsize=8.8, color=GREY)
    fig.subplots_adjust(left=LEFT, right=0.98, top=0.9, bottom=0.12)
    fig.savefig(out)
    return out


if __name__ == "__main__":
    print(main())
