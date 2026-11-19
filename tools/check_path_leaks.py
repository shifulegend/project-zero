#!/usr/bin/env python3
"""Fail when the project-memory docs leak a developer's machine paths or identifiers.

Scans docs/ai/**, the root *.md files and .claude/rules/** for: /home/<name>/ (except the neutral
/home/user/ext used for reference builds), /Users/<name>/, /mnt/..., canonical UUIDs (session
identifiers), and the name of the private readme-guidance repository.

usage: tools/check_path_leaks.py
"""

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCOPE = ("docs/ai/", ".claude/rules/")
PATTERNS = [
    ("home directory path", re.compile(r"/home/(?!user/ext\b)[A-Za-z][\w.-]*/")),
    ("macOS home path", re.compile(r"/Users/[A-Za-z][\w.-]*/")),
    ("mounted drive path", re.compile(r"(?<![\w/])/mnt/[\w.-]+")),
    (
        "UUID (session identifier?)",
        re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"),
    ),
    ("private repository name", re.compile(r"readme-guidance")),
]


def problems(text, path):
    """(line, description) for every machine path, session ID or private name found in `text`."""
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        for label, rx in PATTERNS:
            if rx.search(line):
                out.append(f"{path}:{n}: {label}")
    return out


def in_scope(rel):
    """True for the Markdown files the check covers (root files and the project-memory folders)."""
    return rel.endswith(".md") and (rel.startswith(SCOPE) or "/" not in rel)


def main():
    """Entry point: scans the covered files and returns 1 when anything leaks."""
    files = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).split()
    bad = []
    for rel in files:
        if in_scope(rel):
            bad += problems((ROOT / rel).read_text(encoding="utf-8", errors="replace"), rel)
    for b in bad:
        print(b)
    print("no path leaks" if not bad else f"{len(bad)} possible leak(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
