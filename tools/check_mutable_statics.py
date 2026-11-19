#!/usr/bin/env python3
"""Ratchet on mutable `static` variables in src/ (file-scope or function-local, non-const).

Shared mutable state is what made the sampler and the attention scratch buffers unsafe for parallel
slots. The count per file may not go up (tools/mutable_statics_baseline.txt); lowering it is welcome
(--update). A static that is really per-process (a registry, a lock, a one-time init flag) is allowed
by bumping the baseline in the same commit, which is the review point.

usage: tools/check_mutable_statics.py [--update]
"""

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tools" / "mutable_statics_baseline.txt"
DECL = re.compile(
    r"^\s*static\s+(?!const\b)(?!inline\b)(?!_Thread_local\s+const\b)([^;(){}]*?)\s*(=|;|\[)"
)
SKIP = ("src/api/webui_bundle_generated.c",)


def count(text):
    n = 0
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(("//", "/*", "*")):
            continue
        m = DECL.match(line)
        if m and "const" not in m.group(1).split("*")[-1].split() and "(" not in line.split("=")[0]:
            n += 1
    return n


def counts():
    files = subprocess.check_output(["git", "ls-files", "src"], cwd=ROOT, text=True).split()
    out = {}
    for rel in files:
        if not rel.endswith((".c", ".cpp")) or rel in SKIP:
            continue
        n = count((ROOT / rel).read_text(encoding="utf-8", errors="replace"))
        if n:
            out[rel] = n
    return out


def read_baseline():
    out = {}
    if BASELINE.exists():
        for line in BASELINE.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                k, v = line.rsplit(None, 1)
                out[k] = int(v)
    return out


def compare(cur, base):
    return [
        f"{k}: {v} mutable static(s), baseline {base.get(k, 0)}"
        for k, v in sorted(cur.items())
        if v > base.get(k, 0)
    ]


def main(argv):
    cur = counts()
    if "--update" in argv:
        BASELINE.write_text(
            "# Mutable static variables per file in src/ (tools/check_mutable_statics.py); may only go down.\n"
            + "".join(f"{k} {v}\n" for k, v in sorted(cur.items())),
            encoding="utf-8",
        )
        print(f"baseline written: {sum(cur.values())} in {len(cur)} files")
        return 0
    bad = compare(cur, read_baseline())
    for b in bad:
        print("more mutable statics than the baseline:", b)
    print(f"mutable statics: {sum(cur.values())} in {len(cur)} files")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
