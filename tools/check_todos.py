#!/usr/bin/env python3
"""Every TODO/FIXME in first-party code must carry an ID, TODO(PZ-TODO-<n>) or TODO(QA-ISS-<n>),
and the ID must be listed in docs/ai/todo-register.md.

usage: tools/check_todos.py
"""

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
REGISTER = ROOT / "docs" / "ai" / "todo-register.md"
SUFFIXES = {".c", ".h", ".cpp", ".hpp", ".py", ".sh", ".js", ".svelte"}
SKIP = (
    "include/stb_",
    "src/api/webui_bundle_generated.c",
    "benchmark_results/",
    "tools/check_todos.py",
    "tools/test_check_todos.py",
)
MARK = re.compile(r"\b(TODO|FIXME)\b(\(([A-Za-z0-9-]+)\))?")


def problems(text: str, path: str, registered: set[str]) -> list[str]:
    """Problems in `text`: TODO/FIXME markers with no ID, or an ID that is not in the register."""
    out: list[str] = []
    for n, line in enumerate(text.splitlines(), 1):
        for m in MARK.finditer(line):
            ident = m.group(3)
            if not ident or not re.fullmatch(r"(PZ-TODO|QA-ISS)-\d+", ident):
                out.append(
                    f"{path}:{n}: {m.group(1)} without an ID (write {m.group(1)}(PZ-TODO-<n>) and"
                    " list it)"
                )
            elif ident not in registered:
                out.append(f"{path}:{n}: {ident} is not listed in docs/ai/todo-register.md")
    return out


def registered_ids(text: str) -> set[str]:
    """IDs listed in the register table."""
    return set(re.findall(r"^\|\s*((?:PZ-TODO|QA-ISS)-\d+)\s*\|", text, re.M))


def main() -> int:
    """Entry point: scans the tracked sources against the register and returns 1 on any problem."""
    reg = registered_ids(REGISTER.read_text(encoding="utf-8"))
    files = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).split()
    bad = []
    for rel in files:
        p = pathlib.Path(rel)
        if p.suffix not in SUFFIXES or rel.startswith(SKIP):
            continue
        bad += problems((ROOT / rel).read_text(encoding="utf-8", errors="replace"), rel, reg)
    for b in bad:
        print(b)
    print("todos ok" if not bad else f"{len(bad)} TODO problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
