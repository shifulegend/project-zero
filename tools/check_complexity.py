#!/usr/bin/env python3
"""Ratchet on function size and complexity in the engine: a function in src/ may not exceed
cyclomatic complexity 15 or 100 lines of code unless it is listed in tools/complexity_baseline.txt,
and a listed function may not grow in either number. Split a function, then run --update so its
entry drops or disappears (the list may only shrink). The project targets are lower (20 lines,
complexity 10, docs/ai/engineering-rules.md); this gate stops the worst from getting worse.

Measured with lizard (pip install lizard==1.26.1): C and C++ under src/, generated code excluded.

usage: tools/check_complexity.py [--update]
"""

import csv
import io
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tools" / "complexity_baseline.txt"
MAX_CCN = 10
MAX_NLOC = 100
SKIP = ("src/api/webui_bundle_generated.c",)


def measure() -> dict[str, tuple[int, int]]:
    """{ "file::function": (ccn, nloc) } for every function in src/ (largest wins on a repeated name)."""
    out = subprocess.run(
        [sys.executable, "-m", "lizard", "src", "-l", "c", "-l", "cpp", "--csv"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, tuple[int, int]] = {}
    for row in csv.reader(io.StringIO(out)):
        nloc, ccn, path, name = int(row[0]), int(row[1]), row[6], row[7]
        if path.startswith(SKIP):
            continue
        key = f"{path}::{name}"
        old = result.get(key, (0, 0))
        result[key] = (max(old[0], ccn), max(old[1], nloc))
    return result


def read_baseline() -> dict[str, tuple[int, int]]:
    """Recorded (ccn, nloc) of the existing offenders from tools/complexity_baseline.txt."""
    out = {}
    if BASELINE.exists():
        for line in BASELINE.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                key, ccn, nloc = line.rsplit(None, 2)
                out[key] = (int(ccn), int(nloc))
    return out


def offenders(cur: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    """The functions over either limit."""
    return {k: v for k, v in cur.items() if v[0] > MAX_CCN or v[1] > MAX_NLOC}


def problems(cur: dict[str, tuple[int, int]], base: dict[str, tuple[int, int]]) -> list[str]:
    """Messages for new offenders and for listed ones that grew in complexity or length."""
    out = []
    for key, (ccn, nloc) in sorted(offenders(cur).items()):
        if key not in base:
            out.append(
                f"{key}: complexity {ccn}, {nloc} lines (limits {MAX_CCN} / {MAX_NLOC}); "
                "split it or justify a baseline entry"
            )
            continue
        b_ccn, b_nloc = base[key]
        if ccn > b_ccn or nloc > b_nloc:
            out.append(
                f"{key}: complexity {ccn}, {nloc} lines, grew past its recorded {b_ccn} / {b_nloc}"
            )
    return out


def main(argv: list[str]) -> int:
    """Entry point: writes the baseline with --update, else checks the tree; returns the exit code."""
    cur = measure()
    if "--update" in argv:
        bad = offenders(cur)
        BASELINE.write_text(
            f"# Functions over complexity {MAX_CCN} or {MAX_NLOC} lines (tools/check_complexity.py): "
            "key, complexity, lines; entries may only shrink or go.\n"
            + "".join(f"{k} {v[0]} {v[1]}\n" for k, v in sorted(bad.items())),
            encoding="utf-8",
        )
        print(f"baseline written: {len(bad)} function(s)")
        return 0
    found = problems(cur, read_baseline())
    for line in found:
        print(line)
    print(
        f"functions over the limits: {len(offenders(cur))}"
        if not found
        else f"{len(found)} problem(s)"
    )
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
