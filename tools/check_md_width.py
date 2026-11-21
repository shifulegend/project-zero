"""Markdown prose written or edited from now on is wrapped at 80 columns.

Existing text is left alone (reflowing about 14k lines would bury the history), so only the lines
a change adds are checked: `check_md_width.py [BASE]` compares the working tree with
`git merge-base BASE HEAD` (default origin/master). Exempt: table rows, lines with a URL or an
HTML tag, headings, fenced code, reference definitions, and the append-only logs
(CHANGELOG.md, docs/ai/change-trace.md, docs/ai/mistakes.md, docs/ai/decision-log.md), whose
entries are one line each by convention.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIMIT = 80
LOGS = {
    "CHANGELOG.md",
    "docs/ai/change-trace.md",
    "docs/ai/mistakes.md",
    "docs/ai/decision-log.md",
    "docs/ai/todo-register.md",
}
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def exempt(line: str, in_fence: bool) -> bool:
    """True for a line the width rule does not apply to."""
    text = line.strip()
    return (
        in_fence
        or text.startswith(("|", "#", "<", "[", "```", "~~~"))
        or "http://" in text
        or "https://" in text
        or "](" in text
        or bool(re.search(r"<[a-zA-Z/][^>]*>", text))
    )


def too_wide(path: str, lines: list[str], added: set[int]) -> list[tuple[int, int]]:
    """(line number, width) of every added prose line wider than LIMIT in one file."""
    if path in LOGS:
        return []
    out: list[tuple[int, int]] = []
    in_fence = False
    for number, line in enumerate(lines, 1):
        text = line.rstrip("\n")
        if text.strip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if number in added and len(text) > LIMIT and not exempt(text, in_fence):
            out.append((number, len(text)))
    return out


def added_lines(diff: str) -> dict[str, set[int]]:
    """File -> set of line numbers a unified diff (-U0) adds."""
    result: dict[str, set[int]] = {}
    current = ""
    for row in diff.splitlines():
        if row.startswith("+++ b/"):
            current = row[6:]
            result[current] = set()
        else:
            match = HUNK.match(row)
            if match and current:
                start = int(match.group(1))
                count = int(match.group(2) if match.group(2) is not None else 1)
                result[current].update(range(start, start + count))
    return result


def main(argv: list[str]) -> int:
    """Check the Markdown lines added since BASE; exit 1 when any is too wide."""
    base = argv[1] if len(argv) > 1 else "origin/master"
    merge_base = subprocess.run(
        ["git", "merge-base", base, "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "-U0", merge_base, "--", "*.md"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    problems = 0
    for path, added in sorted(added_lines(diff).items()):
        file = ROOT / path
        if not file.is_file():
            continue
        wide = too_wide(path, file.read_text(encoding="utf-8").splitlines(), added)
        for number, width in wide:
            print(f"{path}:{number}: {width} columns (limit {LIMIT})")
        problems += len(wide)
    print(f"markdown width: {problems} added line(s) over {LIMIT} columns")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
