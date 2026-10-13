"""One number line per probe: medians of look-alike replies, plain answers, OFF and ON (each probe's top writer, frozen
test, B = 16) against its 1% line, with the two yardsticks (OFF to line, plain to line) and the capability slope under
each. Shows why the OFF-referenced score blows up where OFF sits near the line.

    .venv/bin/python analysis/for_james/methodology/make_framings.py
"""

import matplotlib.pyplot as plt
import numpy as np

from make_dots import COLORS, OUT, TASKS, flag_line, strips

# Capability slopes per 10 ECI [95%], two-stage bootstrap: maths/ordering over 14 writers (analysis/post_v8, m45),
# the new tasks over 11 writers (m49 STATUS / frozen_B16_framings.png).
SLOPES = {"maths": ("+0.07", "+0.06"), "ordering": ("+0.13", "+0.10"), "acrostic": ("+0.21", "+0.17"),
          "belief_tracking": ("+0.38", "+0.13"), "contradiction": ("+0.47", "+0.13")}
ORDER = ("maths", "ordering", "acrostic", "belief_tracking", "contradiction")
XLIM = (-12, 14)


def main() -> None:
    fig, axes = plt.subplots(len(ORDER), 1, figsize=(13, 9.5), sharex=True)
    for ax, task in zip(axes, ORDER):
        cfg = TASKS[task]
        line = flag_line(cfg)
        med = {label.split("\n")[0]: float(np.median(xs)) for label, xs, _ in strips(cfg)}
        lookalike, plain, off, on = (med["look-alike replies"], med["plain correct answers"], med["OFF replies"], med["ON replies"])
        ax.axvspan(XLIM[0], line, color="#f6dcd6", zorder=0)
        ax.axvline(line, color=COLORS["line"], lw=2.2)
        ax.hlines(0.42, line, plain, color=COLORS["plain"], lw=1.2, alpha=0.6)
        ax.hlines(-0.42, line, off, color=COLORS["off"], lw=2.4)
        ax.text((line + plain) / 2, 0.52, f"plain to line {plain - line:.1f}", ha="center", va="bottom", fontsize=10, color=COLORS["plain"])
        ax.text((line + off) / 2, -0.52, f"OFF to line {off - line:.1f}", ha="center", va="top", fontsize=10.5,
                color=COLORS["off"], fontweight="bold")
        for x, colour, size in ((lookalike, COLORS["neg"], 140), (plain, COLORS["plain"], 200), (off, COLORS["off"], 200), (on, COLORS["on"], 200)):
            ax.scatter(x, 0, s=size, color=colour, zorder=3)
        ax.annotate("", xy=(on, 0), xytext=(off, 0), arrowprops=dict(arrowstyle="->", color=COLORS["on"], lw=2), zorder=2)
        off_ref, plain_ref = SLOPES[task]
        ax.text(XLIM[1] + 0.4, 0, f"slope from OFF {off_ref}\nslope from plain {plain_ref}", va="center", fontsize=10.5,
                color="#17212b")
        ax.set_title(f"{cfg.title} ({cfg.writer_label})", loc="left", fontsize=11.5, fontweight="bold")
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    axes[-1].set_xlim(*XLIM)
    axes[-1].set_xlabel("probe score (log-odds, prompt in view)")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=9, color=c, label=lab) for lab, c in
               (("look-alike replies", COLORS["neg"]), ("plain answer", COLORS["plain"]), ("OFF", COLORS["off"]), ("ON", COLORS["on"]))]
    handles.append(plt.Line2D([], [], color=COLORS["line"], lw=2.2, label="1% flag line"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=11, bbox_to_anchor=(0.45, -0.01))
    fig.tight_layout(rect=(0, 0.04, 0.86, 1))
    fig.savefig(OUT / "framings.png", dpi=160)


if __name__ == "__main__":
    main()
