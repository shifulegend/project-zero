#!/usr/bin/env python3
"""A small mutation test: do the unit tests notice when one operator in a source file is changed?

For each of a handful of operator sites in FILE (comparison and arithmetic operators outside
comments and strings) the script flips one operator, rebuilds the named test, runs it and records
whether the test failed (the mutant is killed) or still passed (it survived, a gap in the test).
The file is always restored. It reports the kill rate and lists the survivors; it is meant for the
nightly workflow and never blocks a pull request.

usage: tools/mutation_smoke.py FILE TEST [--max N] [--seed S] [--min-kill PERCENT]
       e.g. tools/mutation_smoke.py src/sampling/top_p.c test_top_p --max 12
"""

from __future__ import annotations

import argparse
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SWAPS = {"<=": "<", ">=": ">", "==": "!=", "!=": "==", "<": "<=", ">": ">=", "&&": "||", "||": "&&"}
OPERATOR = re.compile(r"<=|>=|==|!=|&&|\|\||(?<![<>=!-])<(?![<=])|(?<![<>=-])>(?![>=])")


def blank_noise(text: str) -> str:
    """The text with comments and string/char literals replaced by spaces (same length)."""
    return re.sub(
        r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
        lambda m: "".join(c if c == "\n" else " " for c in m.group(0)),
        text,
        flags=re.S,
    )


def sites(text: str) -> list[tuple[int, str]]:
    """(offset, operator) of every mutable operator in the code of `text`."""
    code = blank_noise(text)
    return [(m.start(), m.group(0)) for m in OPERATOR.finditer(code) if m.group(0) in SWAPS]


def mutate(text: str, offset: int, operator: str) -> str:
    """`text` with the operator at `offset` replaced by its opposite."""
    return text[:offset] + SWAPS[operator] + text[offset + len(operator) :]


def run_test(test: str) -> str:
    """'killed' when the test build or run fails, 'survived' when it passes."""
    build = subprocess.run(
        ["make", f"build/tests/{test}", "CC=gcc"], cwd=ROOT, capture_output=True, text=True
    )
    if build.returncode != 0:
        return "killed"  # a mutant that does not even build is noticed by the build
    run = subprocess.run(
        [f"build/tests/{test}"], cwd=ROOT, capture_output=True, text=True, timeout=300
    )
    return "killed" if run.returncode != 0 else "survived"


def main(argv: list[str]) -> int:
    """Run the mutants and print the kill rate; exit 1 only when below --min-kill."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file")
    parser.add_argument("test")
    parser.add_argument("--max", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min-kill", type=float, default=0.0)
    args = parser.parse_args(argv)
    path = ROOT / args.file
    original = path.read_text(encoding="utf-8")
    found = sites(original)
    random.Random(args.seed).shuffle(found)
    chosen = sorted(found[: args.max])
    survivors: list[str] = []
    killed = 0
    try:
        for offset, operator in chosen:
            path.write_text(mutate(original, offset, operator), encoding="utf-8")
            line = original.count("\n", 0, offset) + 1
            if run_test(args.test) == "killed":
                killed += 1
            else:
                survivors.append(f"{args.file}:{line}: {operator} -> {SWAPS[operator]}")
    finally:
        path.write_text(original, encoding="utf-8")
    rate = 100.0 * killed / len(chosen) if chosen else 100.0
    print(
        f"mutation kill rate for {args.file} with {args.test}: {killed}/{len(chosen)} ({rate:.0f}%)"
    )
    for survivor in survivors:
        print("survived:", survivor)
    return 1 if rate < args.min_kill else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
