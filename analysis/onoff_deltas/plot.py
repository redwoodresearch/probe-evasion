"""Attack minus plain ('delta') for off-policy vs on-policy probe pairs, Qwen3.5-27B layer 48.
Numbers from analysis/onoff_pairs/table.md, analysis/fa_cost/fa_cost.json and the m3a2/m3a3/m11 JSONs (verified there).
Valid evasions per attempt, instruction removed, unless noted."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OFF, ON = "#eb6834", "#2a78d6"  # dataviz slots 2 and 1, validated as a pair in analysis/onoff_pairs/plot.py
INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#e1e0d9"

# (label, plain off, attacked off [seeds], plain on, attacked on [seeds]) in % of attempts
SARCASM = [
    ("Each probe's own 5-round search\n1% line", 5.8, [53.6, 52.8, 54.2], 1.2, [45.8, 48.2, 3.8]),
    ("Best prompt for either probe\n1% line", 5.8, [78.8, 79.0, 79.8], 1.2, [69.2, 72.0, 17.4]),
    ("Each probe's own search\n5% line (stricter test of depth)", 0.4, [24.2], 0.2, [2.8]),
]
OBJECTIVE = [
    ("Maths: 'confirm their figure'\n(search aimed at the on-policy probe)", 0.0, [2.6, 1.4, 1.4], 0.0, [0.0, 0.0, 0.6]),
    ("Maths: earlier transferred prompt\n(off-policy = Gemma-written text; instruction visible)", 0.0, [4.3, 3.8, 4.5], 0.0, [0.0, 0.0, 0.0]),
    ("Chemistry: best attack on each", 0.0, [0.0], 0.0, [0.0]),
]
FA = [("Sarcasm: false alarms needed to catch\n90% of its own search's replies", 0.41, [10.8], 0.41, [3.8])]


def panel(ax, rows, xmax, title, unit="pts"):
    for i, (label, p_off, a_off, p_on, a_on) in enumerate(rows):
        y = len(rows) - 1 - i
        for dy, col, p, a, name in ((0.18, OFF, p_off, a_off, "off"), (-0.18, ON, p_on, a_on, "on")):
            d = a[0] - p
            ax.barh(y + dy, d, height=0.32, color=col, edgecolor="white", linewidth=1.5)
            if len(a) > 1:
                ds = [x - p for x in a]
                ax.plot([min(ds), max(ds)], [y + dy] * 2, color=INK, lw=1)
                ax.scatter(ds[1:], [y + dy] * (len(ds) - 1), s=14, color=INK, zorder=3)
            right = max([d, 0] + [x - p for x in a])
            ax.text(right + xmax * 0.015, y + dy, f"+{d:.1f} {unit}  (plain {p:g}% → {a[0]:g}%)",
                    va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=9.5)
    ax.set_xlim(0, xmax)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", colors=MUTED, labelsize=8.5)


fig = plt.figure(figsize=(13, 10.5))
gs = fig.add_gridspec(3, 1, height_ratios=[3, 3, 1.3], hspace=0.5, left=0.33, right=0.97, top=0.86, bottom=0.08)
panel(fig.add_subplot(gs[0]), SARCASM, 105, "Sarcasm: extra evasion caused by the attack (percentage points)")
panel(fig.add_subplot(gs[1]), OBJECTIVE, 6, "Maths and chemistry: extra evasion caused by the attack (percentage points; note the 0–6 scale)")
panel(fig.add_subplot(gs[2]), FA, 16, "Sarcasm: extra false-alarm rate a defender must accept to catch 90% of the attack")
fig.text(0.02, 0.965, "How much does adversarial prompting add, for off-policy vs on-policy probes?", fontsize=15,
         fontweight="bold", color=INK)
fig.text(0.02, 0.935, "Bar = attacked minus plain prompt, same probe, Qwen3.5-27B layer 48, valid evasions per attempt, instruction removed. "
         "Black line and dots: training seeds 1-2 where available.", fontsize=9.5, color=MUTED)
fig.text(0.02, 0.913, "Orange = off-policy probe (trained on Claude-written text)     Blue = on-policy probe (trained on Qwen's own replies)",
         fontsize=10, color=INK)
fig.text(0.02, 0.02, "Pairs are NOT matched in training-data size or composition (matched pairs running in m20). Plain-prompt false-alarm cost is "
         "0.41% for both sarcasm probes, so panel 3's gap is not a head start.", fontsize=8.5, color=MUTED)
for p in ("png", "pdf"):
    fig.savefig(f"share/onoff_deltas.{p}", dpi=170)
