"""Builds index.html from page.src.html: fills the generated diagrams and the probe-training fragments."""

from pathlib import Path

HERE = Path(__file__).parent
N_LAYERS = 64
PROBE_LAYER = 48
FREEZE_ROUNDS = (0, 4, 8, 16)
ECI_RANGE = (130.0, 170.0)
WRITERS = [  # (name, ECI, Anthropic)
    ("Claude Opus 5.5", 167.3, True), ("GPT-6 Astra", 166.4, False), ("GPT-6.1 Sol", 166.1, False),
    ("Claude Sonnet 5.5", 165.0, True), ("Claude Fable 5.1", 164.7, True), ("GPT-6 Sol", 162.7, False),
    ("Claude Opus 4.8", 158.2, True), ("Gemini 3.8 Flash", 156.7, False), ("GPT-5.4 Mini", 148.8, False),
    ("Qwen3.5-397B", 146.7, False), ("Claude Haiku 4.5", 142.4, True), ("Qwen3.5-9B", 139.5, False),
    ("GPT-5 nano", 139.4, False), ("Llama 4 Maverick", 132.2, False),
]


def layers() -> str:
    return "".join(f'<span class="hit" title="layer {i}"></span>' if i == PROBE_LAYER else "<span></span>" for i in range(1, N_LAYERS + 1))


def eci_rows() -> str:
    lo, hi = ECI_RANGE
    rows = []
    for name, eci, anthropic in WRITERS:
        left = 100 * (eci - lo) / (hi - lo)
        dot = f'<span class="dot{" anth" if anthropic else ""}" style="left:{left:.1f}%" title="{eci}"></span>'
        rows.append(f'<div class="row"><span>{name} <span class="note">{eci:.0f}</span></span><div class="bar">{dot}</div></div>')
    ticks = "".join(f"<span>{t}</span>" for t in range(int(lo), int(hi) + 1, 10))
    rows.append(f'<div class="row"><span></span><div class="ax">{ticks}</div></div>')
    return "".join(rows)


def rounds() -> str:
    return "".join(f'<span class="fz">{r}</span>' if r in FREEZE_ROUNDS else f"<span>{r}</span>" for r in range(17))


def main() -> None:
    page = (HERE / "page.src.html").read_text()
    fills = {
        "%%LAYERS%%": layers(), "%%NLAYERS%%": str(N_LAYERS), "%%ECI%%": eci_rows(), "%%ROUNDS%%": rounds(),
        "%%LAYERWHY%%": (HERE / "fragments" / "layer_why.html").read_text(),
        "%%TRAINING%%": (HERE / "fragments" / "training.html").read_text(),
        "%%CHECKS%%": (HERE / "fragments" / "checks.html").read_text(),
    }
    for key, value in fills.items():
        assert page.count(key) == 1, key
        page = page.replace(key, value)
    assert "%%" not in page
    (HERE / "index.html").write_text(page)


if __name__ == "__main__":
    main()
