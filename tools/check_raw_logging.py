#!/usr/bin/env python3
"""Fails when src/ writes a diagnostic to stderr without the logger.

Every diagnostic goes through core/log.h (tn_log, tn_log_out) so that --log-format json gives one
parseable object per line and credentials are redacted in one place. A raw fprintf(stderr, ...),
fputs(..., stderr) or perror() in src/ bypasses that. The files below may write to stderr directly
because they are terminal UI or the logger itself: the in-place progress and live tok/s lines, the
step-timing report, and core/log.c.

usage: tools/check_raw_logging.py
"""

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ALLOWED = {
    "src/core/log.c": "the logger itself",
    "src/core/progress.c": "in-place terminal progress line",
    "src/cli/live_stats.c": "live tok/s status line (ANSI cursor control)",
    "src/cli/repl.c": "newline after the live tok/s line",
    "src/core/step_timing.c": "per-step timing report (a table, not a log)",
    "src/transformer/gen_session.c": "step-timing report call",
    "src/speculative/spec_decode.c": "step-timing report call",
    "src/memory/safe_alloc.c": "flush only",
    "src/cli/session_checks.c": "isatty() test",
}
RAW = re.compile(
    r"\b(fprintf|vfprintf|dprintf)\s*\(\s*stderr\b|\b(fputs|fputc|putc|fwrite)\s*\([^;]*\bstderr\s*\)|\bperror\s*\("
)
COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)


def raw_uses(text):
    """Line numbers of raw stderr writes in `text` (comments ignored)."""

    def blank(m):
        """Replaces a comment with spaces, keeping line breaks."""
        return re.sub(r"[^\n]", " ", m.group(0))

    code = COMMENT.sub(blank, text)
    return [code.count("\n", 0, m.start()) + 1 for m in RAW.finditer(code)]


def main():
    """Entry point: scans src/ and returns 1 when a file outside the allow-list writes to stderr."""
    files = subprocess.check_output(["git", "ls-files", "src"], cwd=ROOT, text=True).split()
    bad = []
    for rel in files:
        if not rel.endswith((".c", ".cpp")) or rel in ALLOWED:
            continue
        text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        for line in raw_uses(text):
            bad.append(f"{rel}:{line}: raw write to stderr; use tn_log() from core/log.h")
    for b in bad:
        print(b)
    print(f"{len(bad)} raw stderr write(s) outside the allow-list")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
