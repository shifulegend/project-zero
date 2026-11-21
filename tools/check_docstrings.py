#!/usr/bin/env python3
"""Ratchet on Python functions without a docstring in tools/ (private helpers and tests included).

The per-file count may not go up (tools/docstring_baseline.txt); documenting a function and lowering
the baseline (--update) is welcome. Every def counts (public or private, methods, nested functions
and the unit tests); dunder methods are the exception. Ruff (D100-D107) covers the public ones in
every Python file, this check also the private ones.

usage: tools/check_docstrings.py [--update]
"""

import ast
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tools" / "docstring_baseline.txt"
HEADER = (
    "# Public functions without a docstring, per file in tools/ (tools/check_docstrings.py);"
    " may only go down.\n"
)


def missing(source: str) -> list[str]:
    """Names of public functions in `source` that have no docstring."""
    out = []
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not (node.name.startswith("__") and node.name.endswith("__"))
            and not ast.get_docstring(node)
        ):
            out.append(node.name)
    return out


def counts() -> dict[str, int]:
    """Returns {relative path: number of undocumented functions} for tools/*.py."""
    files = subprocess.check_output(["git", "ls-files", "tools"], cwd=ROOT, text=True).split()
    result = {}
    for rel in files:
        if not rel.endswith(".py") or "benchmark_results" in rel:
            continue
        n = len(missing((ROOT / rel).read_text(encoding="utf-8")))
        if n:
            result[rel] = n
    return result


def read_baseline(text: str) -> dict[str, int]:
    """Parses the baseline file into {path: count}."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            path, n = line.rsplit(" ", 1)
            out[path] = int(n)
    return out


def compare(current: dict[str, int], baseline: dict[str, int]) -> list[str]:
    """Returns messages for every file that has more undocumented functions than its baseline."""
    return [
        f"{path}: {n} undocumented public function(s), baseline {baseline.get(path, 0)}"
        for path, n in sorted(current.items())
        if n > baseline.get(path, 0)
    ]


def main(argv: list[str]) -> int:
    """Entry point: writes the baseline with --update, else checks the tree; returns the exit code."""
    current = counts()
    if "--update" in argv:
        body = "".join(f"{p} {n}\n" for p, n in sorted(current.items()))
        BASELINE.write_text(HEADER + body, encoding="utf-8")
        print(f"baseline written: {sum(current.values())} in {len(current)} files")
        return 0
    baseline = read_baseline(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
    problems = compare(current, baseline)
    for p in problems:
        print(p)
    print(f"undocumented public functions: {sum(current.values())} in {len(current)} files")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
