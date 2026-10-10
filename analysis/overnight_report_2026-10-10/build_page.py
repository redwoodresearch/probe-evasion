"""Render report.md into a self-styled HTML page (page/index.html) with figures under page/fig/."""
import re
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).parent
FIGS = {"../../missions/m23_model_size/combined_sizes.png": "fig/model_sizes.png",
        "../../missions/m22_score_only_iterations/runs/qwen35_27b/rounds_summary.png": "fig/maths_rounds.png"}

STYLE = """
<title>Overnight Report, Oct 10</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono&display=swap">
<style>
/* one reading column, tables scroll inside their own box */
:root { --bg:#f6f5f0; --paper:#ffffff; --ink:#1b2430; --muted:#5d6673; --rule:#dcdcd4; --accent:#b4471e; --code:#eceae2;
  --serif:'Source Serif 4', Georgia, serif; --sans:'IBM Plex Sans', system-ui, sans-serif; --mono:'IBM Plex Mono', ui-monospace, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#15191f; --paper:#1d232b; --ink:#e8e6df; --muted:#a3abb6; --rule:#323a44; --accent:#f08a5d; --code:#262d36; color-scheme: dark } }
:root[data-theme="dark"] { --bg:#15191f; --paper:#1d232b; --ink:#e8e6df; --muted:#a3abb6; --rule:#323a44; --accent:#f08a5d; --code:#262d36; color-scheme: dark }
body { background: var(--bg); color: var(--ink); font-family: var(--sans); font-size: 16px; line-height: 1.6; margin: 0; padding-inline: 16px; padding-block: 40px 80px; }
main { max-width: 760px; margin: 0 auto; }
h1 { font-family: var(--serif); font-weight: 600; font-size: 2.3rem; line-height: 1.15; margin: 0 0 1.2rem; text-wrap: balance; }
h2 { font-family: var(--serif); font-weight: 600; font-size: 1.45rem; margin: 2.6rem 0 0.8rem; padding-top: 1.2rem; border-top: 1px solid var(--rule); text-wrap: balance; }
p, li { max-width: 68ch; }
strong { font-weight: 600; }
ol > li, ul > li { margin: 0.35rem 0; }
ol > li::marker { font-family: var(--serif); font-weight: 600; color: var(--accent); }
code { font-family: var(--mono); font-size: 0.82em; background: var(--code); padding: 0.1em 0.35em; border-radius: 4px; overflow-wrap: anywhere; }
.table { overflow-x: auto; margin: 1rem 0; background: var(--paper); border: 1px solid var(--rule); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 0.88rem; font-variant-numeric: tabular-nums; }
th, td { text-align: left; vertical-align: top; padding: 0.55rem 0.7rem; border-bottom: 1px solid var(--rule); }
th { font-weight: 600; color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }
tr:last-child td { border-bottom: 0; }
img { max-width: 100%; height: auto; display: block; margin: 1.2rem 0 0.4rem; background: #ffffff; border-radius: 8px; border: 1px solid var(--rule); }
p > em:only-child { display: block; color: var(--muted); font-size: 0.88rem; font-style: normal; }
.lede { font-size: 1.05rem; }
</style>
"""

def main() -> None:
    md = (HERE / "report.md").read_text()
    for src, dst in FIGS.items():
        assert src in md, src
        md = md.replace(src, dst)
    html = markdown.markdown(md, extensions=["tables"])
    html = re.sub(r"<table>", '<div class="table"><table>', html).replace("</table>", "</table></div>")
    out = HERE / "page" / "index.html"
    out.write_text(STYLE + "<main>\n" + html + "\n</main>\n")
    print(out)

if __name__ == "__main__":
    main()
