#!/usr/bin/env python3
"""Every byte the HTTP server sends goes through net_write_all() (src/api/net_io.c).

A bare write() on a socket can stop short or fail with EINTR; net_write_all() loops on both. The
check fails when another file under src/api/ calls write(), send() or fwrite() directly.

usage: tools/check_socket_writes.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOWED = {"net_io.c"}
CALL = re.compile(r"(?<![\w.>])(?:write|send|fwrite)\s*\(")


def violations(text: str) -> list[int]:
    """Line numbers of direct write()/send()/fwrite() calls in the code of `text`."""
    code = re.sub(
        r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\])*\"",
        lambda m: "".join(c if c == "\n" else " " for c in m.group(0)),
        text,
        flags=re.S,
    )
    return [code.count("\n", 0, m.start()) + 1 for m in CALL.finditer(code)]


def main() -> int:
    """Check src/api/*.c; returns the exit code."""
    problems = 0
    for path in sorted((ROOT / "src" / "api").glob("*.c")):
        if path.name in ALLOWED:
            continue
        for line in violations(path.read_text(encoding="utf-8", errors="replace")):
            print(f"{path.relative_to(ROOT)}:{line}: direct write; use net_write_all()")
            problems += 1
    print(f"socket writes: {problems} direct call(s) outside net_io.c")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
