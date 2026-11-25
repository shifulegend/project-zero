#!/usr/bin/env python3
"""Every header under include/ (except vendored stb_*) must use a classic include guard whose name
starts with TN_ and ends with _H (#ifndef / #define / ... / #endif). No #pragma once, no other
prefixes (PZ_, PROJECT_ZERO_, none), no two headers with the same guard. --fix renames a
non-conforming guard to TN_<PATH>_H (include/api/net_io.h -> TN_API_NET_IO_H).

usage: tools/check_header_guards.py [--fix]    (--fix rewrites non-conforming headers)
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKIP = ("stb_",)


def expected(rel: pathlib.PurePath) -> str:
    """The guard name a header must use: TN_ plus its path under include/, upper-cased."""
    return "TN_" + re.sub(r"[^A-Za-z0-9]", "_", rel.as_posix()).upper()


def guard_of(text: str) -> tuple[str, str] | None:
    """(ifndef name, define name) of the header's include guard, or None when it has none."""
    m = re.search(r"^\s*#\s*ifndef\s+(\w+)\s*\n\s*#\s*define\s+(\w+)", text, re.M)
    return (m.group(1), m.group(2)) if m else None


def check(text: str, rel: pathlib.PurePath) -> list[str]:
    """Return a list of problems for one header's text."""
    probs: list[str] = []
    if re.search(r"^\s*#\s*pragma\s+once", text, re.M):
        probs.append("uses #pragma once")
    g = guard_of(text)
    if g is None:
        probs.append("has no #ifndef/#define include guard")
    elif g[0] != g[1]:
        probs.append(f"#ifndef {g[0]} does not match #define {g[1]}")
    elif not re.fullmatch(r"TN_[A-Z0-9_]+_H", g[0]):
        probs.append(f"guard {g[0]} must look like TN_<NAME>_H (e.g. {expected(rel)})")
    return probs


def fix(text: str, rel: pathlib.PurePath) -> str:
    """Returns `text` with its guard (or #pragma once) rewritten to the expected TN_ guard for header `rel`."""
    want = expected(rel)
    g = guard_of(text)
    if re.search(r"^\s*#\s*pragma\s+once\s*\n", text, re.M):
        text = re.sub(
            r"^\s*#\s*pragma\s+once\s*\n",
            f"#ifndef {want}\n#define {want}\n",
            text,
            count=1,
            flags=re.M,
        )
        return text.rstrip("\n") + f"\n\n#endif /* {want} */\n"
    if g and g[0] == g[1]:
        return re.sub(rf"\b{re.escape(g[0])}\b", want, text)
    return text


def main(argv: list[str]) -> int:
    """Entry point: checks every header under include/ (and rewrites them with --fix); returns the exit code."""
    do_fix = "--fix" in argv
    seen: dict[str, pathlib.PurePath] = {}
    bad = 0
    for f in sorted((ROOT / "include").rglob("*.h")):
        if f.name.startswith(SKIP):
            continue
        rel = f.relative_to(ROOT / "include")
        text = f.read_text(encoding="utf-8")
        probs = check(text, rel)
        if probs and do_fix:
            f.write_text(fix(text, rel), encoding="utf-8")
            probs = check(f.read_text(encoding="utf-8"), rel)
        for p in probs:
            print(f"include/{rel}: {p}")
            bad += 1
        g = guard_of(text)
        if g and g[0] in seen and seen[g[0]] != rel:
            print(f"include/{rel}: guard {g[0]} also used by include/{seen[g[0]]}")
            bad += 1
        if g:
            seen[g[0]] = rel
    print("header guards ok" if not bad else f"{bad} header guard problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
