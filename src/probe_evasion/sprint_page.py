"""Render sprint/index.html: tracks, results, live spend and running jobs. Hand-kept content is in sprint/sprint.json;
spend and running jobs are read live at build time.

    uv run python -m probe_evasion.sprint_page
"""

import base64
import html
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

from probe_evasion import config
from probe_evasion.judge import USAGE_LOG

SPRINT = Path("sprint")
OUT = SPRINT / "index.html"
SPRINT_START = "2026-10-06"
# Sonnet-class list price per million tokens (input, output); an estimate, the API does not report cost.
CLAUDE_PRICE = (3.0, 15.0)


def modal_cli(*args: str):
    out = subprocess.run(["uv", "run", "modal", *args, "--json"], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def modal_spend() -> dict:
    """Our app's cost from Modal's billing report (complete days only, so today lags)."""
    ours = lambda rows: sum(float(r["cost"]) for r in rows if r["description"] == config.MODAL_APP)
    return {"sprint": ours(modal_cli("billing", "report", "--start", SPRINT_START)),
            "today": ours(modal_cli("billing", "report", "--for", "today"))}


def openrouter_spend() -> float:
    load_dotenv(".env")
    req = urllib.request.Request("https://openrouter.ai/api/v1/key",
                                 headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return float(json.load(r)["data"]["usage"])


def claude_spend() -> float:
    if not USAGE_LOG.exists():
        return 0.0
    calls = [json.loads(line) for line in USAGE_LOG.read_text().splitlines()]
    return sum(c["input_tokens"] * CLAUDE_PRICE[0] + c["output_tokens"] * CLAUDE_PRICE[1] for c in calls) / 1e6


def running_jobs() -> list[dict]:
    """Our live Modal apps, plus the local processes driving them."""
    apps = [a for a in modal_cli("app", "list") if a["description"] == config.MODAL_APP and a["state"] != "stopped"]
    ps = subprocess.run(["ps", "-eo", "etime=,command="], capture_output=True, text=True, check=True).stdout
    local = [line.strip().split(None, 1) for line in ps.splitlines() if "modal run -m probe_evasion" in line and "caffeinate" not in line]
    by_job = {cmd.split("modal run -m ", 1)[1]: etime for etime, cmd in local}  # uv and its python child both match
    return [{"what": job, "running_for": etime} for job, etime in by_job.items()] or \
        [{"what": f"Modal app {a['app_id']}", "running_for": f"since {a['created_at'][11:16]}"} for a in apps]


def img(path: str) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


STYLE = """
/* Layout: one column of stacked panels; spend tiles and the two tracks sit side by side and stack on phones. */
:root {
  --bg: #f6f7f9; --panel: #ffffff; --ink: #1c2027; --muted: #5d6573; --rule: #dde1e7;
  --accent: #c2410c; --accent-soft: #fdebdc; --good: #1d7f52; --good-soft: #e3f3ea; --run: #2a62c9; --run-soft: #e4ecfa;
  --display: "Bricolage Grotesque", "Avenir Next", system-ui, sans-serif;
  --body: "Atkinson Hyperlegible", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #13161b; --panel: #1b1f26; --ink: #e8eaee; --muted: #9aa3b1; --rule: #2c323c;
  --accent: #f08a4b; --accent-soft: #3a2518; --good: #4cc38a; --good-soft: #16301f; --run: #7aa7f5; --run-soft: #1a2640; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #13161b; --panel: #1b1f26; --ink: #e8eaee; --muted: #9aa3b1; --rule: #2c323c;
  --accent: #f08a4b; --accent-soft: #3a2518; --good: #4cc38a; --good-soft: #16301f; --run: #7aa7f5; --run-soft: #1a2640; color-scheme: dark }
body { background: var(--bg); color: var(--ink); font-family: var(--body); font-size: 15px; line-height: 1.5; }
main { max-width: 1080px; margin: 0 auto; padding-inline: 16px; padding-block: 28px 56px; display: grid; gap: 28px; }
h1, h2, h3 { font-family: var(--display); text-wrap: balance; margin: 0; }
h1 { font-size: 2rem; font-weight: 700; letter-spacing: -0.01em; }
h2 { font-size: 1.15rem; font-weight: 650; }
h3 { font-size: 1.05rem; }
p { margin: 0; max-width: 70ch; }
.lede { font-size: 1.1rem; }
.muted { color: var(--muted); }
.eyebrow { font-family: var(--mono); font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
section { display: grid; gap: 12px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
.tile { background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: 14px 16px; display: grid; gap: 2px; }
.tile .num { font-family: var(--mono); font-size: 1.6rem; font-variant-numeric: tabular-nums; }
.tile.total { border-color: var(--accent); }
.tracks { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
.track { background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: 18px; display: grid; gap: 10px; align-content: start; min-width: 0; }
.track header { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; flex-wrap: wrap; }
.pill { font-family: var(--mono); font-size: 0.75rem; padding: 2px 9px; border-radius: 999px; text-transform: uppercase; letter-spacing: 0.06em; }
.pill.running { background: var(--run-soft); color: var(--run); }
.pill.done { background: var(--good-soft); color: var(--good); }
.pill.blocked { background: var(--accent-soft); color: var(--accent); }
dl { margin: 0; display: grid; grid-template-columns: max-content 1fr; gap: 6px 14px; }
dt { color: var(--muted); font-size: 0.85rem; padding-top: 1px; }
dd { margin: 0; min-width: 0; }
ul.findings { margin: 0; padding-left: 18px; display: grid; gap: 4px; }
.table-wrap { overflow-x: auto; background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th, td { text-align: left; padding: 8px 14px; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-weight: 600; color: var(--muted); font-size: 0.8rem; }
tr:last-child td { border-bottom: 0; }
td.mono { font-family: var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; }
.empty { color: var(--muted); padding: 14px; }
.readings { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }
.reading { border-top: 3px solid var(--accent); padding-top: 8px; display: grid; gap: 4px; }
figure { margin: 0; background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: 14px; display: grid; gap: 8px; }
figure img { width: 100%; height: auto; background: #fff; border-radius: 4px; }
figcaption { color: var(--muted); font-size: 0.9rem; }
"""


def esc(s: str) -> str:
    return html.escape(str(s))


def build() -> str:
    d = json.loads((SPRINT / "sprint.json").read_text())
    modal, openrouter, claude = modal_spend(), openrouter_spend(), claude_spend()
    jobs = running_jobs()
    tiles = [("Modal GPUs, sprint", modal["sprint"], "billing report, complete days"),
             ("Modal GPUs, today", modal["today"], "lags by a few hours"),
             ("Claude judge", claude, "estimate from logged tokens; logging began Oct 7"),
             ("OpenRouter", openrouter, "whole key, all-time")]
    total = modal["sprint"] + claude + openrouter
    tile_html = "".join(f'<div class="tile"><span class="eyebrow">{esc(n)}</span><span class="num">${v:,.0f}</span>'
                        f'<span class="muted">{esc(note)}</span></div>' for n, v, note in tiles)
    tile_html += (f'<div class="tile total"><span class="eyebrow">Total so far</span><span class="num">${total:,.0f}</span>'
                  f'<span class="muted">Modal sprint + Claude + OpenRouter</span></div>')
    job_rows = "".join(f'<tr><td>{esc(j["what"])}</td><td class="mono">{esc(j["running_for"])}</td></tr>' for j in jobs)
    jobs_html = (f'<div class="table-wrap"><table><thead><tr><th>Job</th><th>Running for</th></tr></thead>'
                 f'<tbody>{job_rows}</tbody></table></div>') if jobs else '<div class="table-wrap"><p class="empty">Nothing running.</p></div>'
    tracks = ""
    for t in d["tracks"]:
        findings = "".join(f"<li>{esc(f)}</li>" for f in t["findings"]) or '<li class="muted">No results yet.</li>'
        tracks += (f'<article class="track"><header><h3>Track {esc(t["id"])}: {esc(t["name"])}</h3>'
                   f'<span class="pill {esc(t["status"])}">{esc(t["status"])}</span></header>'
                   f'<dl><dt>Probe</dt><dd>{esc(t["probe"])}</dd><dt>Model</dt><dd>{esc(t["model"])}</dd>'
                   f'<dt>Now</dt><dd>{esc(t["now"])}</dd><dt>Next</dt><dd>{esc(t["next"])}</dd></dl>'
                   f'<span class="eyebrow">Findings</span><ul class="findings">{findings}</ul></article>')
    readings = "".join(f'<div class="reading"><strong>{esc(n)}</strong><span class="muted">{esc(t)}</span></div>' for n, t in d["readings"])
    results = "".join(f'<figure><img src="{img(r["image"])}" alt="{esc(r["title"])}"><figcaption><strong>{esc(r["title"])}.</strong> '
                      f'{esc(r["caption"])}</figcaption></figure>' for r in d["results"]) or \
        '<div class="table-wrap"><p class="empty">Figures appear here as runs finish.</p></div>'
    log_rows = "".join(f'<tr><td class="mono">{esc(a)}</td><td>{esc(b)}</td></tr>' for a, b in reversed(d["log"]))
    return f"""<title>{esc(d["title"])}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{STYLE}</style>
<main>
  <header style="display:grid;gap:8px">
    <span class="eyebrow">Probe evasion · Redwood Research · updated {time.strftime("%Y-%m-%d %H:%M")}</span>
    <h1>{esc(d["title"])}</h1>
    <p class="lede">{esc(d["question"])}</p>
    <p class="muted">{esc(d["why"])}</p>
  </header>
  <section><h2>Results</h2>{results}</section>
  <section><h2>Tracks</h2><div class="tracks">{tracks}</div></section>
  <section><h2>How each reply is scored</h2><div class="readings">{readings}</div><p class="muted">{esc(d["evasion_rule"])}</p></section>
  <section><h2>Running now</h2>{jobs_html}</section>
  <section><h2>Spend</h2><div class="tiles">{tile_html}</div></section>
  <section><h2>Log</h2><div class="table-wrap"><table><tbody>{log_rows}</tbody></table></div></section>
</main>
"""


if __name__ == "__main__":
    OUT.write_text(build())
    print(f"wrote {OUT}")
