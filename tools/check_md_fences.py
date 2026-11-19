#!/usr/bin/env python3
"""Every fenced code block in the Markdown files must name a language (```bash, ```c, ```text).
An untagged fence renders without highlighting and hides what the block is. --fix tags the
untagged ones as `text`; change the tag by hand where a better one exists.

usage: tools/check_md_fences.py [--fix]
"""

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKIP = ("benchmark_results/",)
FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")


def untagged(lines):
    """Line numbers (0-based) of opening fences without an info string."""
    out, fence = [], None
    for i, line in enumerate(lines):
        m = FENCE.match(line)
        if not m:
            continue
        run = m.group(2)
        if fence is None:
            fence = (run[0], len(run))
            if not m.group(3).strip():
                out.append(i)
        elif run[0] == fence[0] and len(run) >= fence[1] and not m.group(3).strip():
            fence = None
    return out


def fix_text(text):
    lines = text.split("\n")
    for i in untagged(lines):
        lines[i] = lines[i].rstrip() + "text"
    return "\n".join(lines)


def main(argv):
    do_fix = "--fix" in argv
    files = subprocess.check_output(["git", "ls-files", "*.md"], cwd=ROOT, text=True).split()
    bad = 0
    for rel in files:
        if rel.startswith(SKIP):
            continue
        p = ROOT / rel
        text = p.read_text(encoding="utf-8", errors="replace")
        n = untagged(text.split("\n"))
        if n and do_fix:
            p.write_text(fix_text(text), encoding="utf-8")
            continue
        for i in n:
            print(f"{rel}:{i + 1}: code fence without a language")
            bad += 1
    print("md fences ok" if not bad else f"{bad} untagged fence(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
