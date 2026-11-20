#!/usr/bin/env python3
"""Ratchet on source lines wider than 100 columns (.editorconfig max_line_length = 100).

clang-format and ruff keep code within 100 columns, so what is left are comments, long strings and
markup they cannot wrap. The number of such lines per file may not go up
(tools/line_width_baseline.txt); wrap a line and run --update to lower it. Generated and vendored
files are skipped.

usage: tools/check_line_width.py [--update]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tools" / "line_width_baseline.txt"
LIMIT = 100
SUFFIXES = (".c", ".h", ".cpp", ".py", ".ts", ".svelte", ".sh")
SKIP_PARTS = ("stb_", "generated", "node_modules", "benchmark_results")
HEADER = "# Lines wider than 100 columns per file (tools/check_line_width.py); may only fall.\n"


def wide_lines(text: str) -> int:
    """Number of lines in `text` longer than LIMIT characters."""
    return sum(1 for line in text.splitlines() if len(line) > LIMIT)


def tracked() -> list[str]:
    """Tracked source files this check looks at."""
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split("\n")
    return [
        name
        for name in out
        if name.endswith(SUFFIXES) and not any(part in name for part in SKIP_PARTS)
    ]


def measure() -> dict[str, int]:
    """{ path: wide lines } for every tracked file that has any."""
    result: dict[str, int] = {}
    for name in tracked():
        path = ROOT / name
        if path.is_file():
            count = wide_lines(path.read_text(encoding="utf-8", errors="replace"))
            if count:
                result[name] = count
    return result


def read_baseline() -> dict[str, int]:
    """The recorded counts."""
    out: dict[str, int] = {}
    if BASELINE.exists():
        for line in BASELINE.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                name, value = line.rsplit(None, 1)
                out[name] = int(value)
    return out


def problems(now: dict[str, int], base: dict[str, int]) -> list[str]:
    """Files with more wide lines than recorded (a new file starts at zero)."""
    return [
        f"{name}: {count} line(s) over {LIMIT} columns, recorded {base.get(name, 0)}"
        for name, count in sorted(now.items())
        if count > base.get(name, 0)
    ]


def main(argv: list[str]) -> int:
    """Write the baseline with --update, else check the tree; returns the exit code."""
    now = measure()
    if "--update" in argv:
        text = HEADER + "".join(f"{k} {v}\n" for k, v in now.items())
        BASELINE.write_text(text, encoding="utf-8")
        print(f"baseline written: {sum(now.values())} line(s) in {len(now)} file(s)")
        return 0
    found = problems(now, read_baseline())
    for line in found:
        print(line)
    print(
        f"lines over {LIMIT} columns: {sum(now.values())}"
        if not found
        else f"{len(found)} problem(s)"
    )
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
