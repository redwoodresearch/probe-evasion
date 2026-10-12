"""Render report.md into page/index.html (figures under page/fig/) with the shared renderer analysis/md_page.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from md_page import render  # noqa: E402

HERE = Path(__file__).parent
FIGS = {"../../missions/m23_model_size/combined_sizes.png": "fig/model_sizes.png",
        "../../missions/m22_score_only_iterations/runs/qwen35_27b/rounds_summary.png": "fig/maths_rounds.png"}


def main() -> None:
    md = (HERE / "report.md").read_text()
    for src, dst in FIGS.items():
        assert src in md, src
        md = md.replace(src, dst)
    out = HERE / "page" / "index.html"
    out.write_text(render(md, "Overnight Report, Oct 10"))
    print(out)


if __name__ == "__main__":
    main()
