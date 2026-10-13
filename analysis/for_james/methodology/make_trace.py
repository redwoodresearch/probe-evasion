"""Per-token probe trace for James's methodology page: raw token scores, the smoothed score, and the reply's peak."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS = Path("missions/m45_toggle_sweep/runs")
OUT = Path("analysis/for_james/methodology/figures")
PROBE = "q27orr@L48@s0"
LINE = -8.06
ALPHA = 0.5  # running average: smoothed_t = 0.5 * token_t + 0.5 * smoothed_(t-1); reply score = max over tokens
ON_REPLY_MARK = "Gia the giraffe and Cal the cat"  # Fable 5.1, chain 0, B = 16, the median ON reply in this chain


def smoothed(xs: list[float]) -> np.ndarray:
    out, state = [], 0.0  # neutral start, as in gepa.ema_logit
    for x in xs:
        state = ALPHA * x + (1 - ALPHA) * state
        out.append(state)
    return np.array(out)


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def main() -> None:
    on = next(r for r in read(RUNS / "d/qwen35_27b_ordering/test_checked/fable51.jsonl")
              if "chain_0@B16#on" in r["key"] and ON_REPLY_MARK in r["reply"])
    plain = next(r for r in read(RUNS / "c/qwen35_27b_ordering/test_checked/plain.jsonl")
                 if r["task"]["id"] == on["task"]["id"] and r["seed"] == on["seed"])
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.4), sharey=True)
    for ax, (title, r) in zip(axes, (("Plain answer", plain), ("ON reply (Fable 5.1's style)", on))):
        tokens, xs = r["tokens_visible"], r["logits_visible"][PROBE]
        sm = smoothed(xs)
        assert abs(sm.max() - r["peaks_visible"][PROBE]) < 0.05, (sm.max(), r["peaks_visible"][PROBE])
        t = np.arange(len(xs))
        ax.bar(t, np.array(xs) - (-16), bottom=-16, color="#c9cfcb", width=0.7, label="each token's raw score")
        ax.plot(t, sm, color="#1f6f5c", lw=2, label="smoothed (running average, α = 0.5)")
        k = int(sm.argmax())
        ax.scatter([k], [sm[k]], s=60, color="#c2541b", zorder=5, label="reply score = highest smoothed point")
        ax.annotate(f"{sm[k]:+.1f}", (k, sm[k]), textcoords="offset points", xytext=(8, 4), color="#c2541b", fontsize=9, fontweight="bold")
        ax.axhline(LINE, color="#b3261e", lw=1.2)
        ax.text(len(xs) - 0.5, LINE + 0.6, "1% flag line", color="#b3261e", fontsize=8, ha="right")
        ax.set_xticks(t, [tok.strip() or "␣" for tok in tokens], rotation=60, fontsize=7, ha="right")
        ax.set_xlim(-0.8, max(len(plain["tokens_visible"]), len(on["tokens_visible"])) - 0.2)
        ax.set_title(f"{title}: “{r['reply'][:90]}{'…' if len(r['reply']) > 90 else ''}”", loc="left", fontsize=9)
        ax.set_ylim(-16, 23)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(fontsize=8, loc="upper right", frameon=False)
    fig.supylabel("ordering probe score (log-odds)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "trace.png", dpi=160)


if __name__ == "__main__":
    main()
