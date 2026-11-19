#!/usr/bin/env python3
"""Ratchet on file length: a first-party source file may not exceed 500 lines unless it is listed in
tools/file_size_baseline.txt, and a listed file may not grow beyond its recorded length. Split a
large file, then run --update so its entry drops or disappears (the list may only shrink).

Counted: .c .h .cpp .hpp under src/ include/ tests/ tools/, .py under tools/ and tests/, and webui/src.
Not counted: vendored stb headers, the generated web UI bundle, and the lines of Python docstrings
(documentation must not push a file over the limit).

usage: tools/check_file_sizes.py [--update]
"""

import ast
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tools" / "file_size_baseline.txt"
LIMIT = 500
SUFFIXES = {".c", ".h", ".cpp", ".hpp", ".py", ".js", ".svelte"}
SCOPE = ("src/", "include/", "tests/", "tools/", "webui/src/")
SKIP = ("include/stb_", "src/api/webui_bundle_generated.c", "tools/screenshots/")


def docstring_lines(source):
    """Number of lines that Python docstrings (module, class and function) occupy in `source`."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0
    total = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                total += first.end_lineno - first.lineno + 1
    return total


def sizes():
    """Line count of every counted tracked source file (Python docstring lines excluded)."""
    files = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).split()
    out = {}
    for rel in files:
        if (
            not rel.startswith(SCOPE)
            or rel.startswith(SKIP)
            or pathlib.Path(rel).suffix not in SUFFIXES
        ):
            continue
        data = (ROOT / rel).read_bytes()
        n = data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)
        if rel.endswith(".py"):
            n -= docstring_lines(data.decode("utf-8", errors="replace"))
        out[rel] = n
    return out


def read_baseline():
    """Recorded sizes of the existing oversized files from tools/file_size_baseline.txt."""
    out = {}
    if BASELINE.exists():
        for line in BASELINE.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                path, n = line.rsplit(None, 1)
                out[path] = int(n)
    return out


def problems(cur, base, limit=LIMIT):
    """Messages for files over the limit that are new, or listed ones that grew past their recorded size."""
    out = []
    for rel, n in sorted(cur.items()):
        allowed = base.get(rel, limit)
        if n > max(allowed, limit if rel not in base else allowed):
            if rel in base:
                out.append(f"{rel}: {n} lines, grew past its recorded {base[rel]}")
            else:
                out.append(
                    f"{rel}: {n} lines (limit {limit}); split it or justify a baseline entry"
                )
    return out


def main(argv):
    """Entry point: writes the baseline with --update, else checks the tree; returns the exit code."""
    cur = sizes()
    if "--update" in argv:
        big = {k: v for k, v in cur.items() if v > LIMIT}
        BASELINE.write_text(
            f"# Files over {LIMIT} lines and their lengths (tools/check_file_sizes.py); entries may only shrink or go.\n"
            + "".join(f"{k} {v}\n" for k, v in sorted(big.items())),
            encoding="utf-8",
        )
        print(f"baseline written: {len(big)} file(s) over {LIMIT} lines")
        return 0
    bad = problems(cur, read_baseline())
    for b in bad:
        print(b)
    over = sum(1 for v in cur.values() if v > LIMIT)
    print(f"files over {LIMIT} lines: {over}" if not bad else f"{len(bad)} file size problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
