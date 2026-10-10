"""Attack minus plain ('delta') for off-policy vs on-policy probe pairs, Qwen3.5-27B layer 48.
Numbers from analysis/onoff_pairs/table.md, analysis/fa_cost/fa_cost.json and the m3a2/m3a3/m11 JSONs (verified there).
Valid evasions per attempt, instruction removed, unless noted."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OFF, ON = "#eb6834", "#2a78d6"  # dataviz slots 2 and 1, validated as a pair in analysis/onoff_pairs/plot.py
INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#e1e0d9"

# (label, plain off, attacked off [seeds], plain on, attacked on [seeds]) in % of attempts
# Sarcasm: m20 MATCHED pair (same 4,807 rows except the author), final critic-reviewed result
# (missions/m20_onoff_pairs/sarcasm/RESULT.md). Plain: 31 vs 15 of 500 missed (seed 0).
SARCASM = [
    ("6 fresh searches, each scored on the\nprobe it was aimed at (pooled, seed 0)", 6.2, [726 / 15], 3.0, [883 / 15]),
    ("5 prompts aimed at neither probe\n(hand-written + starting hint)", 6.2, [286 / 17], 3.0, [230 / 17]),
    ("18 older searches, tuned on near-copies\nof the Claude-written probe", 6.2, [4444 / 90], 3.0, [3842 / 90]),
]
ANGER = [  # m20 matched pair; bars = median over 10 training seeds (16:45 interim); plain ~2.5% / 1.5%
    ("Fresh search aimed at the\nmodel-written probe (median of 10 seeds)", 2.5, [97.6], 1.5, [76.0]),
    ("Fresh search aimed at the\nClaude-written probe (median of 10 seeds)", 2.5, [72.6], 1.5, [30.8]),
    ("Old attack, tuned on a near-copy of\nthe Claude-written probe (seed 0, 200 tries)", 2.5, [60.5], 1.5, [75.0]),
]
# m20 matched pairs (agent A), mean over 10 training draws, full-strength replies only; each search scored on the probe it
# was aimed at. Source: missions/m20_onoff_pairs/{flattery,condescension}/analysis.json, key seed_level.primary.
OTHER_TONE = [
    ("Flattery: fresh search aimed at each\nprobe (gap −6 pts [−16, +4])", 10.9, [69.4], 5.4, [70.0]),
    ("Condescension: search warm-started from\nm11 prompts, aimed at each (gap +71 pts [+62, +77])", 0.0, [70.8], 0.0, [0.0]),
    ("Condescension: cold search from\ndefault seeds, aimed at each", 0.0, [0.0], 0.0, [2.8]),
]
OBJECTIVE = [  # NOT matched (pilot pairs)
    ("Maths: 'confirm their figure'\n(search aimed at the model-written probe)", 0.0, [2.6, 1.4, 1.4], 0.0, [0.0, 0.0, 0.6]),
    ("Maths: earlier transferred prompt\n(off-policy = Gemma-written text; instruction visible)", 0.0, [4.3, 3.8, 4.5], 0.0, [0.0, 0.0, 0.0]),
    ("Chemistry: best attack on each", 0.0, [0.0], 0.0, [0.0]),
]
# 95% task-resampled intervals on the gain over plain, (off, on). None = not available (the sarcasm fair test reports
# only the interval on the off-minus-on difference: -13.7 points [-19.5, -7.9]).
CIS = {
    "Maths: 'confirm their figure'\n(search aimed at the model-written probe)": ((0.6, 5.0), (0.0, 0.0)),
    "Anger: false alarms needed to catch\n90% of the old attack": ((1.9, 6.7), (4.6, 12.0)),
    "Sarcasm: false alarms needed to catch 90%\nof Opus search 1 aimed at the Claude-written probe": ((7.0, 11.1), (11.2, 16.6)),
    "Sarcasm: false alarms needed to catch 90%\nof Opus search 1 aimed at the model-written probe": ((3.8, 7.6), (6.7, 10.0)),
}
FA = [("Sarcasm: false alarms needed to catch 90%\nof Opus search 1 aimed at the Claude-written probe", 0.6, [9.0], 0.5, [13.8]),
      ("Sarcasm: false alarms needed to catch 90%\nof Opus search 1 aimed at the model-written probe", 0.6, [5.3], 0.5, [8.4]),
      ("Anger: false alarms needed to catch 90%\nof the search aimed at the model-written probe", 0.4, [8.6], 0.4, [3.0]),
      ("Anger: false alarms needed to catch 90%\nof the search aimed at the Claude-written probe", 0.4, [2.9], 0.4, [1.9]),
      ("Anger: false alarms needed to catch\n90% of the old attack", 0.4, [5.0], 0.4, [8.3])]


def panel(ax, rows, xmax, title, unit="pts"):
    for i, (label, p_off, a_off, p_on, a_on) in enumerate(rows):
        y = len(rows) - 1 - i
        for dy, col, p, a, name in ((0.18, OFF, p_off, a_off, "off"), (-0.18, ON, p_on, a_on, "on")):
            d = a[0] - p
            ax.barh(y + dy, d, height=0.32, color=col, edgecolor="white", linewidth=1.5)
            ci = CIS.get(label, (None, None))[0 if name == "off" else 1]
            if ci is not None:
                ax.errorbar(d, y + dy, xerr=[[d - ci[0]], [ci[1] - d]], fmt="none", ecolor=INK, elinewidth=1.6, capsize=5, zorder=3)
            if len(a) > 1:
                ds = [x - p for x in a]
                ax.scatter(ds[1:], [y + dy] * (len(ds) - 1), s=46, marker="D", facecolor="white", edgecolor=INK, linewidth=1.2, zorder=4)
            right = max([d, 0] + [x - p for x in a] + ([ci[1]] if ci is not None else []))
            ax.text(right + xmax * 0.015, y + dy, f"+{d:.1f} {unit}  (plain {p:g}% → {a[0]:.1f}%)",
                    va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=9.5)
    ax.set_xlim(0, xmax)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", colors=MUTED, labelsize=8.5)


fig = plt.figure(figsize=(13, 18.5))
gs = fig.add_gridspec(5, 1, height_ratios=[3, 3, 3, 3, 4.2], hspace=0.55, left=0.33, right=0.97, top=0.91, bottom=0.05)
panel(fig.add_subplot(gs[0]), SARCASM, 112, "Sarcasm, matched pair: aimed searches get more past the model-written probe (gap −13.7 pts)")
panel(fig.add_subplot(gs[1]), ANGER, 135, "Anger, matched pair: depends on the attack")
panel(fig.add_subplot(gs[2]), OTHER_TONE, 112, "Flattery and condescension, matched pairs: no gap vs a large gap (mean of 10 training draws)")
panel(fig.add_subplot(gs[3]), OBJECTIVE, 8, "Maths and chemistry (pilot pairs, not matched): note the 0–8 scale")
panel(fig.add_subplot(gs[4]), FA, 45, "Extra false-alarm rate a defender must accept to catch 90% of the attack")
fig.text(0.02, 0.978, "Does it matter who wrote the probe's training examples?", fontsize=15,
         fontweight="bold", color=INK)
fig.text(0.02, 0.952, "Bar = attacked minus plain prompt, same probe, Qwen3.5-27B layer 48, valid evasions per attempt, instruction removed. "
         "\nWhisker = 95% interval from resampling tasks. Hollow diamonds = the same probe retrained with a different random seed (seeds 1 and 2); not an interval.", fontsize=9.5, color=MUTED)
fig.text(0.02, 0.930, "Orange = probe trained on Claude-written examples     Blue = probe trained on the model's own (Qwen's) examples",
         fontsize=10, color=INK)
fig.text(0.02, 0.012, "Sarcasm, anger, flattery and condescension pairs are matched (same training data except the author); maths and chemistry are earlier, unmatched pilot pairs.\n"
         "Sarcasm final; flattery and condescension awaiting a last review; anger fair-test numbers preliminary. Plain-prompt false-alarm cost is about 0.5% for every probe.", fontsize=8.5, color=MUTED)
for p in ("png", "pdf"):
    fig.savefig(f"share/onoff_deltas.{p}", dpi=170)
