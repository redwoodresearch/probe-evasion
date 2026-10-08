"""Dashboard of the parallel evasion missions (missions/dashboard.html). Key numbers are hand-kept in
missions/dashboard.json; each mission's live phase line and last-update time are read from its STATUS.md.

    uv run python -m probe_evasion.missions_page
"""

import html
import json
import re
import time
from pathlib import Path

ROOT = Path("missions")
OUT = ROOT / "dashboard.html"
GROUPS = ["Maths", "Meaning vs wording", "Deception"]
ORDER = {"confirmed": 0, "promising": 1, "running": 2, "negative": 3}
STATES = {"promising": "Promising", "running": "Running", "negative": "No evasion", "confirmed": "Confirmed"}

CSS = """
/* Layout: one column of mission rows grouped by concept; each row = identity | result bars | notes. */
:root {
  --bg: #f6f7f9; --surface: #ffffff; --ink: #1c2230; --muted: #5c6577; --line: #dde1e8;
  --attack: #d9682b; --plain: #8a93a3; --good: #23784b; --run: #3b5b9a; --warn-bg: #fdf0e7; --good-bg: #e7f4ec; --run-bg: #e9eef8;
  --display: "Archivo", "Helvetica Neue", Arial, sans-serif; --body: "Public Sans", "Helvetica Neue", Arial, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #12151c; --surface: #1a1f29; --ink: #e6e9ef; --muted: #9aa3b4; --line: #2c3341;
  --attack: #f08a4b; --plain: #7d8698; --good: #5fc28b; --run: #8fb0ef; --warn-bg: #3a2618; --good-bg: #17331f; --run-bg: #1d2840; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #12151c; --surface: #1a1f29; --ink: #e6e9ef; --muted: #9aa3b4; --line: #2c3341;
  --attack: #f08a4b; --plain: #7d8698; --good: #5fc28b; --run: #8fb0ef; --warn-bg: #3a2618; --good-bg: #17331f; --run-bg: #1d2840; color-scheme: dark }
body { background: var(--bg); color: var(--ink); font: 15px/1.55 var(--body); margin: 0; }
.wrap { max-width: 1080px; margin: 0 auto; padding-inline: 20px; padding-block: 28px 56px; display: grid; gap: 28px; }
header { display: grid; gap: 8px; }
.eyebrow { font: 500 12px var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
h1 { font: 700 30px/1.15 var(--display); margin: 0; text-wrap: balance; letter-spacing: -.01em; }
.lede { margin: 0; max-width: 72ch; color: var(--muted); }
.key { display: flex; gap: 18px; flex-wrap: wrap; font-size: 13px; color: var(--muted); }
.key span { display: inline-flex; align-items: center; gap: 6px; }
.sw { width: 18px; height: 8px; border-radius: 2px; display: inline-block; }
section { display: grid; gap: 10px; }
h2 { font: 600 13px var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin: 0; }
.row { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 16px 18px;
  display: grid; grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr); gap: 14px 24px; }
.row > * { min-width: 0; }
.id { display: grid; gap: 6px; align-content: start; }
.top { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.name { font: 650 17px/1.3 var(--display); margin: 0; }
.pill { font: 600 11px var(--mono); letter-spacing: .05em; text-transform: uppercase; padding: 3px 8px; border-radius: 999px; }
.pill.promising { background: var(--warn-bg); color: var(--attack); }
.pill.running { background: var(--run-bg); color: var(--run); }
.pill.negative, .pill.confirmed { background: var(--good-bg); color: var(--good); }
.q { margin: 0; }
.probe { font: 12.5px var(--mono); color: var(--muted); }
.res { display: grid; gap: 8px; align-content: start; }
.bar { display: grid; grid-template-columns: 96px minmax(0, 1fr) 104px; gap: 10px; align-items: center; font-size: 13px; }
.bar .lbl { color: var(--muted); white-space: nowrap; }
.track { height: 10px; background: var(--bg); border-radius: 5px; overflow: hidden; }
.fill { height: 100%; border-radius: 5px; }
.num { font: 600 13px var(--mono); text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
.basis { font-size: 12px; color: var(--muted); }
.pending { font: 13px var(--mono); color: var(--muted); padding-block: 6px; }
.notes { grid-column: 1 / -1; display: grid; gap: 4px; border-top: 1px solid var(--line); padding-top: 10px; font-size: 14px; }
.notes b { font-weight: 600; }
.live { font: 12px var(--mono); color: var(--muted); }
@media (max-width: 720px) { .row { grid-template-columns: 1fr; } .bar { grid-template-columns: 90px minmax(0, 1fr) 96px; } }
"""


def esc(x) -> str:
    return html.escape(str(x))


def live_phase(mission_id: str) -> tuple[str, str]:
    status = ROOT / mission_id / "STATUS.md"
    if not status.exists():
        return "no status file yet", ""
    line = next((l.strip() for l in status.read_text().splitlines() if l.strip().startswith("**Phase:**")), "")
    phase = re.sub(r"\*\*Phase:\*\*\s*", "", line)
    return phase, time.strftime("%-I:%M %p", time.localtime(status.stat().st_mtime))


def bars(m: dict) -> str:
    if not m.get("plain"):
        return '<div class="pending">No numbers yet</div>'
    out = ""
    for label, pair, colour in (("Told to evade", m.get("attack"), "var(--attack)"), ("Not told", m["plain"], "var(--plain)")):
        if pair is None:
            out += f'<div class="bar"><span class="lbl">{label}</span><div class="track"></div><span class="num">a few</span></div>'
            continue
        k, n = pair
        share = k / n if n else 0
        out += (f'<div class="bar"><span class="lbl">{label}</span><div class="track"><div class="fill" style="width:{100 * share:.1f}%;'
                f'background:{colour}"></div></div><span class="num">{100 * share:.0f}% · {k}/{n}</span></div>')
    return out + f'<div class="basis">{esc(m["basis"])}</div>'


def row(m: dict) -> str:
    phase, when = live_phase(m["id"])
    return f"""<article class="row">
  <div class="id">
    <div class="top"><span class="pill {esc(m['state'])}">{esc(STATES[m['state']])}</span><h3 class="name">{esc(m['name'])}</h3></div>
    <p class="q">{esc(m['question'])}</p>
    <div class="probe">{esc(m['probe'])}</div>
  </div>
  <div class="res">{bars(m)}</div>
  <div class="notes">
    <div>{esc(m['note'])}</div>
    <div><b>Next:</b> {esc(m['next'])}</div>
    <div class="live">Agent's own status ({esc(when)}): {esc(phase[:260])}</div>
  </div>
</article>"""


def render() -> str:
    d = json.loads((ROOT / "dashboard.json").read_text())
    ms = d["missions"]
    counts = {s: sum(m["state"] == s for m in ms) for s in STATES}
    sections = "".join(
        f'<section><h2>{esc(g)}</h2>{"".join(row(m) for m in sorted((m for m in ms if m["group"] == g), key=lambda m: ORDER[m["state"]]))}</section>' for g in GROUPS)
    return f"""<title>Evasion Missions</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700&family=Public+Sans:wght@400;600&family=JetBrains+Mono:wght@400;500;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
  <header>
    <div class="eyebrow">Probe evasion sprint · updated {time.strftime('%-I:%M %p')}</div>
    <h1>Evasion missions: where a prompted model gets past a probe</h1>
    <p class="lede">{len(ms)} agents in parallel, each testing one way a probe-based monitor might be evaded by prompting alone,
    while the answer still does the task. {counts['promising']} promising, {counts['running']} running.
    Every number compares the evasion prompt with the plain prompt on the same probe and problems.</p>
    <div class="key"><span><i class="sw" style="background:var(--attack)"></i>Told to evade</span>
      <span><i class="sw" style="background:var(--plain)"></i>Not told (plain prompt)</span></div>
  </header>
  {sections}
</div>
"""


if __name__ == "__main__":
    OUT.write_text(render())
    print(f"wrote {OUT}")
