#!/usr/bin/env python3
"""Require a written justification on a pull request that carries the `size-override` label.

The label skips the 400-line limit (tools/check_pr_size.py), so it must be an explained exception, not
a shortcut: the PR description needs a "Size override justification" section that says why the change
cannot be split into smaller, stacked pull requests. Placeholder text, an HTML comment and a section
that is too short do not count.

usage: PR_BODY="$text" tools/check_size_override.py        (the body is read from the environment)
       tools/check_size_override.py --file body.md
"""

import os
import re
import sys

HEADING = re.compile(r"^#{2,4}\s*size[- ]override justification\s*$", re.IGNORECASE | re.MULTILINE)
NEXT_HEADING = re.compile(r"^#{1,4}\s+\S", re.MULTILINE)
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
MIN_CHARS = 30
PLACEHOLDERS = ("todo", "tbd", "n/a", "na", "none", "...", "explain here", "fill in")


def justification(body):
    """The text under the 'Size override justification' heading, or None when there is no such section."""
    m = HEADING.search(body or "")
    if not m:
        return None
    rest = body[m.end() :]
    nxt = NEXT_HEADING.search(rest)
    text = rest[: nxt.start()] if nxt else rest
    return COMMENT.sub("", text).strip()


def problems(body):
    """Reasons the description does not justify the override (empty list when it does)."""
    text = justification(body)
    if text is None:
        return ['the description has no "Size override justification" section']
    if text.strip().lower().strip(".") in PLACEHOLDERS:
        return ["the justification is a placeholder"]
    if len(re.sub(r"\s+", " ", text)) < MIN_CHARS:
        return [
            f"the justification is shorter than {MIN_CHARS} characters; say why the change cannot be split"
        ]
    return []


def main(argv):
    """Entry point: checks the PR body (from --file or $PR_BODY); returns 1 when it does not justify the label."""
    if "--file" in argv:
        with open(argv[argv.index("--file") + 1], encoding="utf-8") as f:
            body = f.read()
    else:
        body = os.environ.get("PR_BODY", "")
    found = problems(body)
    for p in found:
        print(f"size-override: {p}")
    if not found:
        print("size-override: justification present")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
