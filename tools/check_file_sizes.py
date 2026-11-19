#!/usr/bin/env python3
"""Ratchet on file length: a first-party source file may not exceed 500 lines unless it is listed in
tools/file_size_baseline.txt, and a listed file may not grow beyond its recorded length. Split a
large file, then run --update so its entry drops or disappears (the list may only shrink).

Counted: .c .h .cpp under src/ include/ tests/ tools/, .py under tools/ and tests/, and webui/src.
Not counted: vendored stb headers and the generated web UI bundle.

usage: tools/check_file_sizes.py [--update]
"""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tools" / "file_size_baseline.txt"
LIMIT = 500
SUFFIXES = {".c", ".h", ".cpp", ".py", ".js", ".svelte"}
SCOPE = ("src/", "include/", "tests/", "tools/", "webui/src/")
SKIP = ("include/stb_", "src/api/webui_bundle_generated.c", "tools/screenshots/")


def sizes():
    files = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).split()
    out = {}
    for rel in files:
        if (
            not rel.startswith(SCOPE)
            or rel.startswith(SKIP)
            or pathlib.Path(rel).suffix not in SUFFIXES
        ):
            continue
        with open(ROOT / rel, "rb") as f:
            out[rel] = sum(1 for _ in f)
    return out


def read_baseline():
    out = {}
    if BASELINE.exists():
        for line in BASELINE.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                path, n = line.rsplit(None, 1)
                out[path] = int(n)
    return out


def problems(cur, base, limit=LIMIT):
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
