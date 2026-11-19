#!/usr/bin/env python3
"""Fails when a secret-looking variable is passed to a printing call in src/.

A log line is read by whoever can read the terminal or the log file, so a credential that reaches
printf/fprintf/puts/perror ends up there. The check looks at every print statement in src/ (from the
call to its closing `;`), drops the string literals (help text may name `--api-key`), and flags the
identifiers that hold credentials: api_key, auth_header, authorization, bearer, password,
passphrase, watermark_key. Add a name to SECRET_NAMES when a new credential variable appears.

usage: tools/check_log_secrets.py
"""

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SECRET_NAMES = re.compile(
    r"\b(api_?key|auth_?header|authorization|bearer|password|passphrase|watermark_key|wm_key)\b",
    re.IGNORECASE,
)
PRINT_CALL = re.compile(r"\b(f?printf|vfprintf|dprintf|puts|fputs|perror|snprintf_stderr)\s*\(")
STRING = re.compile(r'"(?:\\.|[^"\\])*"')
COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)
SKIP = ("src/api/webui_bundle_generated.c",)


def strip(text):
    """Blanks comments and string literals but keeps line numbers."""

    def blank(m):
        return re.sub(r"[^\n]", " ", m.group(0))

    return STRING.sub(blank, COMMENT.sub(blank, text))


def violations(text):
    """Returns (line, identifier) for every secret name used inside a print statement."""
    code = strip(text)
    out = []
    for call in PRINT_CALL.finditer(code):
        end = code.find(";", call.end())
        stmt = code[call.end() : end if end != -1 else len(code)]
        for m in SECRET_NAMES.finditer(stmt):
            line = code.count("\n", 0, call.end() + m.start()) + 1
            out.append((line, m.group(0)))
    return out


def main():
    files = subprocess.check_output(["git", "ls-files", "src"], cwd=ROOT, text=True).split()
    bad = []
    for rel in files:
        if not rel.endswith((".c", ".cpp")) or rel in SKIP:
            continue
        for line, name in violations((ROOT / rel).read_text(encoding="utf-8", errors="replace")):
            bad.append(f"{rel}:{line}: '{name}' is passed to a print call")
    for b in bad:
        print(b)
    print(f"log secrets: {len(bad)} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
