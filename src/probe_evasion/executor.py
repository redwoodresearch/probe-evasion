"""Run untrusted programs against stdin. Only call `check` inside the network-blocked Modal
function in modal_app.py, never locally."""

import math
import os
import subprocess
import sys
import tempfile


def run_program(code: str, stdin: str, timeout: float) -> dict:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w") as f:
            f.write(code)
        try:
            p = subprocess.run(
                [sys.executable, path],
                input=stdin,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
                cwd=d,
                env={"PATH": os.environ["PATH"], "LANG": "C.UTF-8"},  # hide container env/tokens
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "stdout": ""}
    if p.returncode != 0:
        return {"status": "error", "stdout": p.stdout[-2000:], "stderr": p.stderr[-2000:]}
    return {"status": "ok", "stdout": p.stdout}


def outputs_match(a: str, b: str, tol: float = 1e-6) -> bool:
    """Whitespace-insensitive token comparison with float tolerance, as in APPS grading."""
    ta, tb = a.split(), b.split()
    if ta == tb:
        return True
    if len(ta) != len(tb):
        return False
    for x, y in zip(ta, tb):
        if x == y:
            continue
        try:
            fx, fy = float(x), float(y)
        except ValueError:
            return False
        # Non-finite values (nan, inf) only match as identical tokens, handled above.
        if not (math.isfinite(fx) and math.isfinite(fy)) or abs(fx - fy) > tol * max(1.0, abs(fy)):
            return False
    return True


def check(job: dict) -> dict:
    """job: {code, tests: [[stdin, expected], ...], probes: [stdin, ...], timeout}.
    Runs tests until the first failure, then runs every probe input."""
    timeout = job["timeout"]
    passed, first_failure = 0, None
    for i, (stdin, expected) in enumerate(job["tests"]):
        r = run_program(job["code"], stdin, timeout)
        if r["status"] == "ok" and outputs_match(r["stdout"], expected):
            passed += 1
            continue
        first_failure = {"test_index": i, **r, "expected": expected[-2000:]}
        break
    probes = [run_program(job["code"], s, timeout) for s in job["probes"]]
    return {
        "tests_passed": passed,
        "tests_total": len(job["tests"]),
        "first_failure": first_failure,
        "probe_runs": probes,
    }
