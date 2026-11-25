#!/usr/bin/env python3
"""Ratchet on warning suppressions in first-party code: the number of NOLINT, noqa and
#pragma ... diagnostic markers must not go up (tools/suppressions_baseline.txt). A new suppression
needs a baseline bump in the same commit (so a reviewer sees it), and a NOLINT must name its check.

usage: tools/check_suppressions.py [--update]
"""

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tools" / "suppressions_baseline.txt"
KINDS = {
    "NOLINT": re.compile(r"\bNOLINT(NEXTLINE)?\b"),
    "noqa": re.compile(r"#\s*noqa\b"),
    "pragma-diagnostic": re.compile(r"#\s*pragma\s+(GCC|clang)\s+diagnostic"),
}
SUFFIXES = {".c", ".h", ".cpp", ".hpp", ".py"}
SKIP = (
    "include/stb_",
    "src/api/webui_bundle_generated.c",
    "benchmark_results/",
    "tools/check_suppressions.py",
    "tools/test_check_suppressions.py",
)


def count(text: str) -> dict[str, int]:
    """Number of NOLINT, noqa and pragma-diagnostic suppressions in `text`, by kind."""
    return {k: len(rx.findall(text)) for k, rx in KINDS.items()}


def bare(text: str, kind: str) -> list[int]:
    """Lines with a blanket NOLINT (it must name its check) or a noqa with no reason after its codes."""
    out: list[int] = []
    for n, line in enumerate(text.splitlines(), 1):
        if kind == "NOLINT" and re.search(r"NOLINT(NEXTLINE)?(?!\()", line):
            out.append(n)
        # a noqa must say why: text after the rule codes ("# noqa: F401  (side-effect import)")
        if kind == "noqa" and re.search(
            r"#\s*noqa\b(?::\s*[A-Z]+[0-9]+(?:\s*,\s*[A-Z]+[0-9]+)*)?\s*$", line
        ):
            out.append(n)
    return out


def totals() -> tuple[dict[str, int], list[str]]:
    """Suppression counts over all tracked first-party files, and the lines of blanket suppressions."""
    files = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).split()
    total = dict.fromkeys(KINDS, 0)
    bare_lines: list[str] = []
    for rel in files:
        if pathlib.Path(rel).suffix not in SUFFIXES or rel.startswith(SKIP):
            continue
        text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        for k, v in count(text).items():
            total[k] += v
        bare_lines += [f"{rel}:{n}" for n in bare(text, "NOLINT") + bare(text, "noqa")]
    return total, bare_lines


def read_baseline() -> dict[str, int]:
    """Baseline counts from tools/suppressions_baseline.txt."""
    out = {}
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            k, v = line.split()
            out[k] = int(v)
    return out


def compare(current: dict[str, int], baseline: dict[str, int]) -> list[str]:
    """Messages for every kind whose count exceeds its baseline."""
    return [
        f"{k}: {current[k]} (baseline {baseline.get(k, 0)})"
        for k in KINDS
        if current[k] > baseline.get(k, 0)
    ]


def main(argv: list[str]) -> int:
    """Entry point: writes the baseline with --update, else checks the tree; returns the exit code."""
    cur, bare_lines = totals()
    if "--update" in argv:
        BASELINE.write_text(
            "# Suppression counts that must not go up (tools/check_suppressions.py).\n"
            + "".join(f"{k} {v}\n" for k, v in cur.items()),
            encoding="utf-8",
        )
        print("baseline written:", cur)
        return 0
    bad = compare(cur, read_baseline())
    for n in bare_lines:
        print(f"{n}: suppression without a check name or a reason")
    for b in bad:
        print("more suppressions than the baseline:", b)
    print(f"suppressions: {cur}")
    return 1 if (bad or bare_lines) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
