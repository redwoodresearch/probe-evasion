"""Rebuild the post's numbers and figures from the saved runs, then copy the figures the post uses into mainline/figures.

CPU only, a few minutes. It needs the raw sweep outputs (missions/m45_toggle_sweep/runs, missions/m49_new_task_toggle/runs
and m45's analysis.json / analysis_d.json), which are too large for git; see mainline/README.md for how they are made.

    .venv/bin/python mainline/make_all.py                 # numbers, then figures
    .venv/bin/python mainline/make_all.py --figures-only  # figures from the committed numbers.json
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
POST = REPO / "analysis" / "post_v9"
PY = str(REPO / ".venv" / "bin" / "python")
RAW_INPUTS = [REPO / "missions/m45_toggle_sweep/runs", REPO / "missions/m49_new_task_toggle/runs",
              REPO / "missions/m45_toggle_sweep/analysis.json", REPO / "missions/m45_toggle_sweep/analysis_d.json"]
FIGURE_SCRIPTS = ["fig_method.py", "fig_2x2.py", "fig_dots.py", "fig_cost.py", "fig_cost_checks.py", "fig_data_sets.py"]
TFIDF_SCRIPTS = ["analysis/tfidf_baseline/for_james/make_figure.py", "analysis/tfidf_baseline/for_james/make_original_figure.py"]
# published name -> source, in the order they appear in the post
FIGURES = {
    "1_method.png": POST / "figures/method_search_score.png",
    "2_main_result.png": POST / "figures/trend_2x2_combined_bands_gold0_key_top.png",
    "3_crossings.png": POST / "figures/dots_2x2.png",
    "4_accuracy_cost.png": POST / "figures/cost_2x2.png",
    "appendix_data_sets.png": POST / "figures/data_sets.png",
    "appendix_passes_vs_cost.png": POST / "figures/pass_vs_cost.png",
    "appendix_cost_by_search.png": POST / "figures/cost_by_search.png",
    "appendix_word_classifier_test.png": REPO / "analysis/tfidf_baseline/for_james/original_test.png",
    "appendix_matched_test.png": REPO / "analysis/tfidf_baseline/for_james/matched_test.png",
}


def run(args: list[str], pythonpath_src: bool = False) -> None:
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")} if pythonpath_src else None
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=REPO, env=env, check=True)


def numbers() -> None:
    with tempfile.TemporaryDirectory() as scratch:
        run([PY, "analysis/post_v9/numbers/extract.py", scratch])
        run([PY, "analysis/post_v9/numbers/compute.py", scratch], pythonpath_src=True)
    run([PY, "analysis/post_v9/numbers/verify.py"])
    run([PY, "analysis/post_v9/numbers/make_tables.py"])


def figures() -> None:
    for script in FIGURE_SCRIPTS:
        run([PY, f"analysis/post_v9/{script}"])
    for script in TFIDF_SCRIPTS:
        run([PY, script])
    run([PY, "analysis/post_v9/appendix_samples/make_samples.py"], pythonpath_src=True)
    out = REPO / "mainline" / "figures"
    out.mkdir(exist_ok=True)
    for name, source in FIGURES.items():
        shutil.copyfile(source, out / name)
    print(f"copied {len(FIGURES)} figures to {out.relative_to(REPO)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures-only", action="store_true", help="skip recomputing numbers.json")
    args = parser.parse_args()
    missing = [str(p.relative_to(REPO)) for p in RAW_INPUTS if not p.exists()]
    if missing:
        sys.exit(f"Missing raw sweep outputs (not in git): {missing}. See mainline/README.md, 'Data that is not in git'.")
    if not args.figures_only:
        numbers()
    figures()


if __name__ == "__main__":
    main()
