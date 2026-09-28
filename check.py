"""Every check in one command: the tests, the notebook pairs, the numbers in the
report and the READMEs, and that reports/REPORT.pdf is what the code and data give.

    .venv/bin/python check.py

It exits with an error if anything fails. The number checks need data/, which
the notebooks build; on a copy without it they are skipped, and it says so.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import jupytext

ROOT = Path(__file__).resolve().parent
failures: list[str] = []


def run_tests() -> None:
    print("tests")
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        r = subprocess.run([sys.executable, str(path)], cwd=ROOT, capture_output=True, text=True)
        passed = sum(line.startswith(("ok ", "PASS ")) for line in r.stdout.splitlines())
        print(f"  {path.name}: {passed} passed" + ("" if r.returncode == 0 else ", then one failed"))
        if r.returncode:
            failures.append(f"{path.name}: {(r.stderr.strip().splitlines() or ['failed'])[-1]}")


def content(py_text: str) -> list[str]:
    """A light-format notebook's lines, without its header and the cell markers jupytext may add."""
    lines = py_text.splitlines()
    if lines and lines[0] == "# ---":
        lines = lines[lines.index("# ---", 1) + 1:]
    return [line.rstrip() for line in lines if line not in ("# +", "# -")]


def check_notebooks() -> None:
    print("notebooks")
    for py in sorted((ROOT / "notebooks").glob("*.py")):
        nb = jupytext.read(py.with_suffix(".ipynb"))
        problems = []
        if content(jupytext.writes(nb, fmt="py:light")) != content(py.read_text()):
            problems.append("the .ipynb does not match the .py")
        code = [c for c in nb.cells if c.cell_type == "code"]
        if [c.get("execution_count") for c in code] != list(range(1, len(code) + 1)):
            problems.append("not run top to bottom")
        if any(o.get("output_type") == "error" for c in code for o in c.get("outputs", [])):
            problems.append("an error in its outputs")
        print(f"  {py.stem}: {', '.join(problems) or 'ok'}")
        failures.extend(f"{py.stem}: {p}" for p in problems)


def check_numbers() -> None:
    print("report and README numbers")
    if not (ROOT / "data" / "processed").exists():
        print("  skipped: there is no data/ here (the notebooks build it)")
        return
    r = subprocess.run([sys.executable, "reports/build_report.py", "--check"], cwd=ROOT,
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("  "):
            print(line)
    if r.returncode:
        err = r.stderr.strip().splitlines()
        first = next((i for i, line in enumerate(err) if line.startswith("AssertionError")), max(len(err) - 5, 0))
        print("  " + "\n  ".join(err[first:]))
        failures.append("report or README numbers (see above)")


if __name__ == "__main__":
    run_tests()
    check_notebooks()
    check_numbers()
    print()
    if failures:
        print(f"{len(failures)} failed:")
        print("\n".join(f"  {f}" for f in failures))
        sys.exit(1)
    print("everything checks out")
