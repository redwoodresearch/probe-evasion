"""Stop hook: don't let a turn end while the sprint page is behind the results.

Blocks (once; never twice in a row) when
  - a results file or status/status.json is newer than status/status.html  -> rebuild the page, or
  - status/status.html is newer than status/.published                     -> republish the artifact
    (after publishing, `touch status/.published`).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "status" / "status.html"
SOURCE = ROOT / "status" / "status.json"
PUBLISHED = ROOT / "status" / ".published"
RESULT_PATTERNS = ("summary*.json", "*_scores.json", "text_monitor_compare.json", "exploit_validity.jsonl",
                   "backdoor_style.jsonl", "examples_scored*.jsonl")


def mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def main() -> None:
    if json.load(sys.stdin).get("stop_hook_active"):
        return  # already blocked once this stop; don't loop
    results = [p for pattern in RESULT_PATTERNS for p in (ROOT / "results").rglob(pattern)]
    newest = max(results + [SOURCE], key=mtime)
    if mtime(newest) > mtime(PAGE):
        reason = (f"The sprint page is behind the results ({newest.relative_to(ROOT)} changed after status/status.html "
                  "was built). Update status/status.json if any claim changed, run `uv run python -m "
                  "probe_evasion.status_page`, republish it with the Artifact tool, then `touch status/.published`.")
    elif mtime(PAGE) > mtime(PUBLISHED):
        reason = ("status/status.html was rebuilt but not republished. Publish it with the Artifact tool (same file "
                  "path), then `touch status/.published`.")
    else:
        return
    print(json.dumps({"decision": "block", "reason": reason}))


if __name__ == "__main__":
    main()
