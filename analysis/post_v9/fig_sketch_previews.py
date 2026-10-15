"""Hand-drawn previews of the post's main figures, to compare with the clean versions (same code, drawn under
sketch.sketch()).

    .venv/bin/python analysis/post_v9/fig_sketch_previews.py   # writes figures/*_sketch.png
"""

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    sketch, method, trend, dots, cost = (load(n) for n in ("sketch", "fig_method", "fig_2x2", "fig_dots", "fig_cost"))
    with sketch.sketch():
        method.search_and_score("method_search_score_sketch.png")
        meta = {r["writer"]: r for r in trend.F.writer_rows(trend.F.load_m45(), "maths")}
        trend.figure(("B16", "B0"), meta, "trend_main_sketch.png", uncertainty="bands_gold0", round0_bars=True, key="top")
        dots.figure(dots.draw_dots, "dots_2x2_sketch.png")


if __name__ == "__main__":
    main()
