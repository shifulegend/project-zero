#!/usr/bin/env python3
"""Fail a pull request that changes more than LIMIT lines (additions + deletions).

Counts `git diff --numstat BASE...HEAD`, ignoring generated, vendored and data files
(EXCLUDE below). A reviewer cannot read a bigger change as one unit: split it, or, for a
change that really must go in one piece (a reformat, a rename, a dependency or framework upgrade,
a data migration, an overall clean-up), have a maintainer add the label "size-override" -- CI then
skips this check and instead requires the PR description to carry a filled
"Size override justification" section (tools/check_size_override.py). Noise that is predictable
(lockfiles, generated code, snapshots, vendored files, translations, fixtures) never counts, so the
label is not needed for it (EXCLUDE below).

usage: tools/check_pr_size.py BASE_REF [--limit N] [--numstat FILE]   (--numstat is for the tests)
"""

import fnmatch
import pathlib
import subprocess
import sys

LIMIT = 400
EXCLUDE = [
    "*package-lock.json",
    "*.lock",
    "src/api/webui_bundle_generated.c",
    "include/stb_*.h",
    "benchmark_results/*",
    "*.svg",
    "*.png",
    "*.jpg",
    "*.gguf",
    "*.snap",
    "*/__snapshots__/*",
    "*.min.js",
    "*.po",
    "*.pot",
    "*/translations/*",
    "tests/fixtures/*",
    "*.golden",
]


def excluded(path: str) -> bool:
    """True when a changed path does not count towards the size limit (lockfiles, generated and vendored files)."""
    return any(fnmatch.fnmatch(path, pat) for pat in EXCLUDE)


def count(numstat_text: str) -> tuple[int, list[tuple[int, str]]]:
    """Return (lines_counted, per_file) from `git diff --numstat` output."""
    total = 0
    per: list[tuple[int, str]] = []
    for line in numstat_text.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        add, dele, path = parts
        if add == "-" or dele == "-":  # binary
            continue
        if excluded(path):
            continue
        n = int(add) + int(dele)
        total += n
        per.append((n, path))
    return total, sorted(per, reverse=True)


def main(argv: list[str]) -> int:
    """Entry point: sums the changed lines against the base ref and returns 1 over the limit."""
    if len(argv) < 2:
        sys.exit(__doc__)
    limit = LIMIT
    if "--limit" in argv:
        limit = int(argv[argv.index("--limit") + 1])
    if "--numstat" in argv:
        text = pathlib.Path(argv[argv.index("--numstat") + 1]).read_text(encoding="utf-8")
    else:
        text = subprocess.check_output(["git", "diff", "--numstat", f"{argv[1]}...HEAD"], text=True)
    total, per = count(text)
    print(f"changed lines (generated/vendored/data files excluded): {total} (limit {limit})")
    if total > limit:
        print("This pull request is too large to review as one unit. Largest files:")
        for n, p in per[:8]:
            print(f"  {n:6d}  {p}")
        print(
            'Split it, or ask a maintainer to add the label "size-override" for a mechanical'
            " change."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
