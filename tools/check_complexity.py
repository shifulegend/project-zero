#!/usr/bin/env python3
"""Ratchet on function size, complexity and nesting: a function in src/ (C and C++), tools/ or tests/
(C and Python) may not exceed cyclomatic complexity 10, 100 lines of code or 3 levels of nested
control structures (lizard's NS) unless it is listed in tools/complexity_baseline.txt, and a listed
function may not grow in any of the three numbers. Split a function,
then run --update so its entry drops or disappears (the list may only shrink). The project target for
function length is lower (20 lines, docs/ai/engineering-rules.md); this gate stops the worst from
getting worse.

Measured with lizard (pip install lizard==1.26.1): generated code excluded.

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
MAX_NS = 3
SKIP = ("src/api/webui_bundle_generated.c",)
SCOPE = ("src", "tools", "tests")  # C and C++ under src/, Python under tools/ and tests/


Triple = tuple[int, int, int]  # (complexity, lines of code, nesting depth)


def measure() -> dict[str, Triple]:
    """{ "file::function": (ccn, nloc, ns) } for every function in src/, tools/ and tests/ (largest wins on a repeated name)."""
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "lizard",
            *SCOPE,
            "-l",
            "c",
            "-l",
            "cpp",
            "-l",
            "python",
            "-Ens",
            "--csv",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    result: dict[str, Triple] = {}
    for row in csv.reader(io.StringIO(out)):
        nloc, ccn, path, name, ns = int(row[0]), int(row[1]), row[6], row[7], int(row[-1])
        if path.startswith(SKIP):
            continue
        key = f"{path}::{name}"
        old = result.get(key, (0, 0, 0))
        result[key] = (max(old[0], ccn), max(old[1], nloc), max(old[2], ns))
    return result


def read_baseline() -> dict[str, Triple]:
    """Recorded (ccn, nloc, ns) of the existing offenders from tools/complexity_baseline.txt."""
    out = {}
    if BASELINE.exists():
        for line in BASELINE.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                key, ccn, nloc, ns = line.rsplit(None, 3)
                out[key] = (int(ccn), int(nloc), int(ns))
    return out


def offenders(cur: dict[str, Triple]) -> dict[str, Triple]:
    """The functions over any of the limits."""
    return {k: v for k, v in cur.items() if v[0] > MAX_CCN or v[1] > MAX_NLOC or v[2] > MAX_NS}


def problems(cur: dict[str, Triple], base: dict[str, Triple]) -> list[str]:
    """Messages for new offenders and for listed ones that grew in complexity, length or nesting."""
    out = []
    for key, (ccn, nloc, ns) in sorted(offenders(cur).items()):
        if key not in base:
            out.append(
                f"{key}: complexity {ccn}, {nloc} lines, nesting {ns} "
                f"(limits {MAX_CCN} / {MAX_NLOC} / {MAX_NS}); split it or justify a baseline entry"
            )
            continue
        b_ccn, b_nloc, b_ns = base[key]
        if ccn > b_ccn or nloc > b_nloc or ns > b_ns:
            out.append(
                f"{key}: complexity {ccn}, {nloc} lines, nesting {ns}, grew past its recorded "
                f"{b_ccn} / {b_nloc} / {b_ns}"
            )
    return out


def main(argv: list[str]) -> int:
    """Entry point: writes the baseline with --update, else checks the tree; returns the exit code."""
    cur = measure()
    if "--update" in argv:
        bad = offenders(cur)
        BASELINE.write_text(
            f"# Functions over complexity {MAX_CCN}, {MAX_NLOC} lines or nesting {MAX_NS} "
            "(tools/check_complexity.py): key, complexity, lines, nesting; entries may only shrink "
            "or go.\n" + "".join(f"{k} {v[0]} {v[1]} {v[2]}\n" for k, v in sorted(bad.items())),
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
