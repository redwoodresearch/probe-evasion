"""30-second summary of the prompt ablations (m6): per prompt, plain vs core-only vs full vs minimal prompt, with what
does the work. Valid evasions out of 200 attempts, instruction hidden, 1% line (missions/m6_prompt_ablation/analysis.json).
-> share/ablation_summary.png"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "share/ablation_summary.png"
A = json.loads((REPO / "missions/m6_prompt_ablation/analysis.json").read_text())
INK, MUTED, GRID = "#1b1f25", "#6b7380", "#e4e7ec"
COLS = {"plain": "#8a929e", "core": "#e0a84a", "full": "#cf3f36", "min": "#7a1f19"}
PANELS = [
    ("sarcasm_r0", "Sarcasm: “That is good news,\nI was just hoping to…”", "min core+examples+bans (round 2)", (250, 140),
     ["Template line alone: nothing (below plain).", "Bans (no exaggeration) push replies flat;",
      "examples keep them sarcastic."]),
    ("sarcasm_r1", "Sarcasm: “How thoughtful of…”\n(vs the patched probe)", "min core+style+examples (round 2)", (253, 128),
     ["The technique line alone gets 90/200.", "Examples teach the exact form;", "a length cap keeps it short."]),
    ("gemma_maths", "Gemma maths, layer 6:\n“the altimeter log shows…”", "min U3+U4+U5+U7+U14+examples (round 2)", (238, 127),
     ["One phrasing does it: an instrument", "“shows” the number. Mentioning the", "probe: no effect (−1 point)."]),
]


def main() -> None:
    fig = plt.figure(figsize=(14, 7), dpi=180)
    for j, (key, title, minimal, words, notes) in enumerate(PANELS):
        ax = fig.add_axes((0.07 + j * 0.315, 0.36, 0.26, 0.42))
        v = A[key]["variants"]
        full2 = v["full (round 2)"]["evaded_hidden@0.01"]
        bars = [("plain\nprompt", v["plain"], "plain"), ("core line\nonly", v["core only"], "core"),
                ("full\nprompt", v["full"], "full"), (f"minimal\n(full on same\nsamples: {full2})", v[minimal], "min")]
        for i, (name, x, c) in enumerate(bars):
            k = x["evaded_hidden@0.01"]
            ax.bar(i, k / 2, 0.7, color=COLS[c])
            ax.text(i, k / 2 + 2, f"{k}", ha="center", fontsize=10.5, weight="bold", color=INK)
        ax.set_xticks(range(4), [b[0] for b in bars], fontsize=8.3)
        ax.set_title(title, fontsize=11, weight="bold", color=INK, loc="left")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if j == 0:
            ax.set_ylabel("% of 200 attempts that get past\n(valid, 1% line, instruction hidden)", fontsize=8.8)
        fig.text(0.07 + j * 0.315, 0.2, "\n".join(notes), fontsize=9.3, color=INK, va="top", linespacing=1.5)
        loss = "small loss on fresh problems (24 vs 29 of 300)" if key == "gemma_maths" else "no real loss"
        fig.text(0.07 + j * 0.315, 0.075, f"{words[0]} words → {words[1]} words, {loss}", fontsize=9, color=COLS["min"],
                 va="top", weight="bold")
    fig.text(0.02, 0.975, "What makes the searched prompts work? A short recipe, not a trick.", fontsize=15.5, weight="bold", color=INK, va="top")
    fig.text(0.02, 0.928, "Each prompt was cut into pieces and re-run with pieces removed. Removing any single piece rarely matters, because "
             "several pieces push toward the same kind of reply.\nAbout half of each prompt can go. Numbers above bars are counts out of 200; "
             "'core line' = the template or technique line plus “must still be sarcastic / get it right”.",
             fontsize=9.2, color=MUTED, va="top", linespacing=1.45)
    fig.savefig(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
