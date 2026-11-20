#!/usr/bin/env python3
"""Ratchet on long boolean conditions in src/: an `if` or `while` condition with four or more
operands (three or more `&&` / `||` joining them) is hard to read and to test, so the number per
file may not go up (tools/bool_chains_baseline.txt). Split the condition into named predicates or
a helper, then run --update; the list may only shrink.

usage: tools/check_bool_chains.py [--update]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tools" / "bool_chains_baseline.txt"
MIN_OPERATORS = 3
HEADER = (
    "# if/while conditions with 4+ operands per file (tools/check_bool_chains.py); may only fall.\n"
)
KEYWORD = re.compile(r"\b(?:if|while)\s*\(")


def strip(text: str) -> str:
    """The source with comments and string/char literals blanked out (same length)."""
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        two = text[i : i + 2]
        if two == "//":
            j = text.find("\n", i)
            j = n if j < 0 else j
        elif two == "/*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
        elif text[i] in "\"'":
            quote, j = text[i], i + 1
            while j < n and text[j] != quote:
                j += 2 if text[j] == "\\" else 1
            j += 1
        else:
            i += 1
            continue
        for k in range(i, min(j, n)):
            if out[k] != "\n":
                out[k] = " "
        i = j
    return "".join(out)


def condition_operators(code: str, start: int) -> int:
    """Number of top-level && and || in the parenthesised condition opening at `start`."""
    depth, count, i = 0, 0, start
    while i < len(code):
        char = code[i]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                break
        elif depth == 1 and code[i : i + 2] in ("&&", "||"):
            count += 1
            i += 1
        i += 1
    return count


def long_chains(text: str) -> int:
    """How many if/while conditions in `text` have MIN_OPERATORS or more boolean operators."""
    code = strip(text)
    return sum(
        1
        for match in KEYWORD.finditer(code)
        if condition_operators(code, match.end() - 1) >= MIN_OPERATORS
    )


def measure() -> dict[str, int]:
    """{ relative path: long chains } for the C and C++ files under src/ that have any."""
    result: dict[str, int] = {}
    for path in sorted((ROOT / "src").rglob("*")):
        if path.suffix in (".c", ".cpp") and path.name != "webui_bundle_generated.c":
            count = long_chains(path.read_text(encoding="utf-8", errors="replace"))
            if count:
                result[str(path.relative_to(ROOT))] = count
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
    """Files whose count is above the recorded one (a new file counts from zero)."""
    return [
        f"{name}: {count} long boolean condition(s), recorded {base.get(name, 0)}"
        for name, count in sorted(now.items())
        if count > base.get(name, 0)
    ]


def main(argv: list[str]) -> int:
    """Write the baseline with --update, else check the tree; returns the exit code."""
    now = measure()
    if "--update" in argv:
        BASELINE.write_text(
            HEADER + "".join(f"{k} {v}\n" for k, v in now.items()), encoding="utf-8"
        )
        print(f"baseline written: {sum(now.values())} condition(s) in {len(now)} file(s)")
        return 0
    found = problems(now, read_baseline())
    for line in found:
        print(line)
    print(
        f"long boolean conditions: {sum(now.values())}" if not found else f"{len(found)} problem(s)"
    )
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
