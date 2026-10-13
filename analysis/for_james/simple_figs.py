"""Two oversimplified figures for James: where replies sit (medians), and the capability trend (ordering frozen test)."""
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 15, "font.family": "DejaVu Sans"})
INK, MUTED, LINE, ON, OFF, PLAIN = "#1b2430", "#7a828c", "#c0392b", "#d9622b", "#2b6cb0", "#555555"

# ---- figure 1: medians on one number line per task (Opus 5.5, practice set; strips/medians.md)
TASKS = {"Maths": {"line": -6.53, "nontask": -9.92, "plain": 10.14, "off": 8.18, "on": 3.94},
         "Ordering": {"line": -8.06, "nontask": -10.14, "plain": 11.44, "off": 6.83, "on": -1.45}}
fig, axes = plt.subplots(2, 1, figsize=(12, 5.2), sharex=True)
for ax, (task, v) in zip(axes, TASKS.items()):
    ax.axvspan(-14, v["line"], color="#f6dcd6", zorder=0)
    ax.axvline(v["line"], color=LINE, lw=2.5)
    ax.text(v["line"] - 0.3, 0.62, "flagged →\n← missed", color=LINE, ha="right", va="center", fontsize=12)
    # label offsets (x, y, alignment): OFF and plain sit close in maths, so their labels lean apart
    for key, colour, label, dx, dy, ha in [("nontask", MUTED, "ordinary\nnon-task replies", 0, -0.42, "center"),
                                           ("plain", PLAIN, "plain answer", -0.3, 0.42, "left"),
                                           ("off", OFF, "attack OFF", 0.3, 0.42, "right"), ("on", ON, "attack ON", 0, 0.42, "center")]:
        ax.scatter(v[key], 0, s=260, color=colour, zorder=3)
        ax.text(v[key] + dx, dy, label, ha=ha, va="center", fontsize=12.5, color=colour)
    ax.annotate("", xy=(v["off"], -0.12), xytext=(v["plain"], -0.12), arrowprops=dict(arrowstyle="->", color=OFF, lw=2))
    ax.annotate("", xy=(v["on"], -0.12), xytext=(v["off"], -0.12), arrowprops=dict(arrowstyle="->", color=ON, lw=2.5))
    ax.text((v["on"] + v["off"]) / 2, -0.36, f"switch: {v['on'] - v['off']:+.1f}", ha="center", color=ON, fontsize=12.5, fontweight="bold")
    ax.text((v["off"] + v["plain"]) / 2, -0.62, f"template in view: {v['off'] - v['plain']:+.1f}", ha="center", color=OFF, fontsize=11.5)
    ax.set_ylim(-0.75, 0.85); ax.set_yticks([]); ax.set_title(task, loc="left", fontsize=15, fontweight="bold")
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
axes[1].set_xlabel("probe score (log-odds); higher = 'doing the task'")
axes[1].set_xlim(-14, 14)
fig.suptitle("Best writer (Claude Opus 5.5): the switch moves replies a lot, but they stay far from the flag line",
             fontsize=14.5, x=0.01, ha="left")
fig.tight_layout(); fig.savefig("analysis/for_james/figures/simple_where.png", dpi=150)
plt.close(fig)

# ---- figure 2: ordering frozen test, OFF-referenced score by writer capability
W = [("Llama 4 Maverick", 132.2, 0.09), ("GPT-5 nano", 139.4, 0.11), ("Qwen3.5-9B", 139.5, 0.12), ("Haiku 4.5", 142.4, 0.25),
     ("Qwen3.5-397B", 146.7, 0.19), ("GPT-5.4 mini", 148.8, 0.09), ("Gemini 3.8 Flash", 156.7, 0.25), ("Opus 4.8", 158.2, 0.40),
     ("GPT-6 Sol", 162.7, 0.35), ("Sonnet 5.5", 165.0, 0.45), ("Opus 5.5", 167.3, 0.58)]
fig, ax = plt.subplots(figsize=(12, 5.2))
xs = [w[1] for w in W]; ys = [w[2] for w in W]
import numpy as np
b, a = np.polyfit(xs, ys, 1); gx = np.array([131, 169])
ax.plot(gx, a + b * gx, ls="--", color=MUTED, lw=2)
ax.scatter(xs, ys, s=150, color=ON, zorder=3)
for name, x, y in W:
    if name in ("Opus 5.5", "Sonnet 5.5", "Opus 4.8", "GPT-6 Sol", "Llama 4 Maverick", "Gemini 3.8 Flash"):
        ax.annotate(name, (x, y), xytext=(-8, 10), textcoords="offset points", ha="right", fontsize=12)
ax.axhline(1, color=LINE, lw=2.5); ax.text(131, 0.95, "attack ON reaches the flag line", color=LINE, va="top", fontsize=12.5)
ax.set_ylim(0, 1.05); ax.set_xlim(130, 169)
ax.set_xlabel("prompt-writing model capability (Epoch Capabilities Index)")
ax.set_ylabel("how far ON gets from OFF\ntoward the flag line")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_title("Stronger writers get further, but none past halfway (ordering, fresh problems)", loc="left", fontsize=14.5)
fig.tight_layout(); fig.savefig("analysis/for_james/figures/simple_trend.png", dpi=150)
print("ok")
