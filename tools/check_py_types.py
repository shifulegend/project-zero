#!/usr/bin/env python3
"""Ratchet on Python typing: every tracked .py file must pass `mypy --strict` (mypy.ini), except the
files listed in tools/py_types_baseline.txt with their current error counts. A listed file may not get
more errors, and a file that became clean must be removed from the list (it may only shrink).

Not checked: benchmark_results/ (archived run output, moving out of the tree).

usage: tools/check_py_types.py [--update]
"""

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tools" / "py_types_baseline.txt"
SKIP = ("benchmark_results/",)
ERROR = re.compile(r"^(?P<path>[^:\n]+\.py):\d+(?::\d+)?: error: ", re.MULTILINE)


def tracked_python() -> list[str]:
    """Tracked .py files that are type-checked."""
    files = subprocess.check_output(["git", "ls-files", "*.py"], cwd=ROOT, text=True).split()
    return [f for f in files if not f.startswith(SKIP)]


def error_counts(files: list[str]) -> dict[str, int]:
    """Number of mypy --strict errors per file (files with no errors are absent)."""
    run = subprocess.run(
        [sys.executable, "-m", "mypy", "--config-file", "mypy.ini", *files],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    counts: dict[str, int] = {}
    for m in ERROR.finditer(run.stdout):
        counts[m.group("path")] = counts.get(m.group("path"), 0) + 1
    return counts


def read_baseline() -> dict[str, int]:
    """Recorded error counts of the not-yet-typed files."""
    out: dict[str, int] = {}
    if BASELINE.exists():
        for line in BASELINE.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                path, n = line.rsplit(None, 1)
                out[path] = int(n)
    return out


def problems(cur: dict[str, int], base: dict[str, int]) -> list[str]:
    """Messages for new type errors and for baseline entries that are now clean."""
    out = []
    for path, n in sorted(cur.items()):
        allowed = base.get(path, 0)
        if n > allowed:
            out.append(f"{path}: {n} mypy --strict error(s), baseline allows {allowed}")
    for path in sorted(base):
        if path not in cur:
            out.append(f"{path}: now clean; remove it from tools/py_types_baseline.txt (--update)")
    return out


def main(argv: list[str]) -> int:
    """Entry point: writes the baseline with --update, else checks the tree; returns the exit code."""
    cur = error_counts(tracked_python())
    if "--update" in argv:
        BASELINE.write_text(
            "# Python files that do not pass mypy --strict yet (tools/check_py_types.py): path, errors.\n"
            "# Entries may only shrink or go.\n"
            + "".join(f"{p} {n}\n" for p, n in sorted(cur.items())),
            encoding="utf-8",
        )
        print(f"baseline written: {len(cur)} file(s)")
        return 0
    found = problems(cur, read_baseline())
    for line in found:
        print(line)
    print(f"files not yet strict-typed: {len(cur)}" if not found else f"{len(found)} problem(s)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
