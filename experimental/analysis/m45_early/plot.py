"""Early practice-set scores from m45 batch c (from the agent's 12 Oct report), against writer ECI, coloured by OFF leakage."""
import matplotlib.pyplot as plt
import numpy as np

ECI = {"Opus 5.5": 167.3, "Sonnet 5.5": 165.0, "GPT-6 Sol": 162.7, "Opus 4.8": 158.2, "Gemini 3.8 Flash": 156.7,
       "GPT-5.4 Mini": 148.8, "Qwen3.5-397B": 146.7, "Haiku 4.5": 142.4, "Qwen3.5-9B": 139.5, "GPT-5 nano": 139.4,
       "Llama 4 Maverick": 132.2}
# writer: (score, OFF replies still showing the style, of 80); None = not reported
ORDERING = {"Opus 5.5": (0.30, 0), "Sonnet 5.5": (0.34, 0), "GPT-6 Sol": (0.22, 2), "Opus 4.8": (0.33, 0),
            "Gemini 3.8 Flash": (0.11, 26), "GPT-5.4 Mini": (0.06, 27), "Qwen3.5-397B": (0.15, 19), "Haiku 4.5": (0.19, 16),
            "Qwen3.5-9B": (0.07, 50), "GPT-5 nano": (0.06, 74), "Llama 4 Maverick": (0.04, 49)}
MATHS = {"Opus 5.5": (0.19, 21), "Sonnet 5.5": (0.12, 35), "GPT-6 Sol": (0.08, 59), "Opus 4.8": (0.26, 8),
         "Gemini 3.8 Flash": (0.06, 45), "GPT-5.4 Mini": (0.04, None), "Qwen3.5-397B": (0.01, 76), "Haiku 4.5": (0.04, 46),
         "Qwen3.5-9B": (0.00, 77), "GPT-5 nano": (0.06, 79), "Llama 4 Maverick": (0.01, 36)}
OFFSET = {"Qwen3.5-9B": (6, -12), "GPT-5 nano": (6, 6), "Opus 4.8": (6, 4), "Gemini 3.8 Flash": (6, -12)}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
cmap = plt.get_cmap("viridis_r")
for ax, (title, data) in zip(axes, [("Ordering five people", ORDERING), ("Maths word problems", MATHS)]):
    x = np.array([ECI[w] for w in data]); y = np.array([data[w][0] for w in data])
    slope, icept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min() - 1, x.max() + 1, 2)
    ax.plot(xs, icept + slope * xs, ls="--", color="#888888", lw=1.2, label=f"least-squares line ({slope * 10:+.3f} per 10 ECI)")
    for w, (s, leak) in data.items():
        colour = "#bbbbbb" if leak is None else cmap(leak / 80)
        ax.scatter(ECI[w], s, s=90, color=colour, edgecolor="#333333", lw=0.6, zorder=3)
        ax.annotate(w, (ECI[w], s), xytext=OFFSET.get(w, (6, 4)), textcoords="offset points", fontsize=8.5)
    ax.axhline(0, color="#444444", lw=0.8)
    ax.set_title(title, fontsize=12, loc="left")
    ax.set_xlabel("Prompt-writing model capability (Epoch Capabilities Index)")
    ax.grid(color="#eeeeee"); ax.legend(loc="upper left", fontsize=9, frameon=False)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
axes[0].set_ylabel("Switch score: ON vs OFF drop,\nshare of the plain-to-line gap (1 = flag line)")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 80)); sm.set_array([])
cb = fig.colorbar(sm, ax=axes, shrink=0.85, pad=0.01); cb.set_label("OFF replies still showing the attack style (of 80)")
fig.suptitle("Matched ON/OFF switch sweep on Qwen3.5-27B: EARLY practice-set numbers (not the final test)\n"
             "Each dot: one writer's mean over 3 searches and its last 3 rounds (fast writers at round 5–8). Read with the instruction in view; correct pairs only.",
             fontsize=11, x=0.02, ha="left")
fig.subplots_adjust(top=0.82, right=0.86, wspace=0.08)
fig.savefig("experimental/analysis/m45_early/early_switch_scores.png", dpi=160)
print("experimental/analysis/m45_early/early_switch_scores.png")
