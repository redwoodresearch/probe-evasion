"""Stop hook: don't let a turn end while the sprint page is behind the results.

Blocks (once; never twice in a row) when
  - a results file or sprint/sprint.json is newer than sprint/page.html  -> rebuild the page, or
  - sprint/page.html is newer than sprint/.published                     -> republish the artifact
    (after publishing, `touch sprint/.published`).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "sprint" / "page.html"
SOURCE = ROOT / "sprint" / "sprint.json"
PUBLISHED = ROOT / "sprint" / ".published"
RESULT_DIRS = ("gepa", "concept_probes")  # the optimized-prompt phase; earlier pages are frozen
RESULT_PATTERNS = ("summary*.json", "calibration.json", "candidates.jsonl", "gemma.json", "qwen35.json")


def mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def main() -> None:
    if json.load(sys.stdin).get("stop_hook_active"):
        return  # already blocked once this stop; don't loop
    results = [p for d in RESULT_DIRS for pattern in RESULT_PATTERNS for p in (ROOT / "results" / d).rglob(pattern)]
    newest = max(results + [SOURCE], key=mtime)
    if mtime(newest) > mtime(PAGE):
        reason = (f"The sprint page is behind the results ({newest.relative_to(ROOT)} changed after sprint/page.html "
                  "was built). Update sprint/sprint.json if any claim changed, run `uv run python -m "
                  "probe_evasion.sprint_page`, republish it with the Artifact tool, then `touch sprint/.published`.")
    elif mtime(PAGE) > mtime(PUBLISHED):
        reason = ("sprint/page.html was rebuilt but not republished. Publish it with the Artifact tool (same file "
                  "path), then `touch sprint/.published`.")
    else:
        return
    print(json.dumps({"decision": "block", "reason": reason}))


if __name__ == "__main__":
    main()
