#!/usr/bin/env python3
"""No disabled code in first-party C/C++: no `#if 0` blocks, no commented-out statements.

Dead code rots silently and misleads readers; version control already keeps old code. The check
fails on an `#if 0` line and on a `//` comment line that is a C statement (a call or an
assignment ending in `;`). Vendored stb_* headers are skipped. Baseline is zero, so any hit is new.

usage: tools/check_dead_code.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIRS = ("src", "include", "tests")
SUFFIXES = {".c", ".h", ".cpp", ".hpp"}
VENDORED = ("stb_",)
IF_ZERO = re.compile(r"^\s*#\s*if\s+0\b")
STATEMENT = re.compile(r"^\s*//\s*(?:[\w\->.\[\]*]+\s*=[^=][^;]*|[\w\->.]+\([^;]*\));\s*$")


def violations(text: str) -> list[tuple[int, str]]:
    """(line, reason) for each `#if 0` or commented-out statement in `text`."""
    found = []
    for number, line in enumerate(text.splitlines(), 1):
        if IF_ZERO.match(line):
            found.append((number, "#if 0 block"))
        elif STATEMENT.match(line):
            found.append((number, "commented-out statement"))
    return found


def main() -> int:
    """Scan the source directories; returns the exit code."""
    problems = 0
    for directory in DIRS:
        for path in sorted((ROOT / directory).rglob("*")):
            if path.suffix not in SUFFIXES or path.name.startswith(VENDORED):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for line, reason in violations(text):
                print(f"{path.relative_to(ROOT)}:{line}: {reason}")
                problems += 1
    print(f"dead code: {problems} hit(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
