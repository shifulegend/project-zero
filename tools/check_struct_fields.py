#!/usr/bin/env python3
"""Ratchet on struct width: a struct declared in include/ may not have more than 25 direct fields
unless it is listed in tools/struct_fields_baseline.txt, and a listed struct may not gain fields. Split
a wide struct into cohesive sub-structs, then run --update so its entry drops (the list may only shrink).

Counted: every `typedef struct [tag] { ... } Name;` in include/**/*.h; a nested struct or union counts as
the one field it declares. A declaration with several names (`int a, b;`) counts each name.

usage: tools/check_struct_fields.py [--update]
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tools" / "struct_fields_baseline.txt"
LIMIT = 25
SKIP = ("include/stb_",)  # vendored single-file libraries
TYPEDEF = re.compile(r"typedef\s+struct\s*\w*\s*\{")
COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)


def body_end(text: str, start: int) -> int:
    """Index of the `}` that closes the brace opened just before `start`."""
    depth = 1
    i = start
    while depth:
        depth += (text[i] == "{") - (text[i] == "}")
        i += 1
    return i - 1


def count_fields(body: str) -> int:
    """Direct fields in a struct body (nested braces collapsed; `int a, b;` counts two)."""
    flat = []
    depth = 0
    for ch in body:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif depth == 0:
            flat.append(ch)
    total = 0
    for decl in "".join(flat).split(";"):
        decl = decl.strip()
        if not decl:
            continue
        # a function pointer `void (*cb)(int)` is one field; commas inside parentheses do not split
        no_parens = re.sub(r"\([^()]*\)", "()", decl)
        total += no_parens.count(",") + 1
    return total


def widths() -> dict[str, int]:
    """Field count of every struct typedef under include/, keyed `file::Name`."""
    out: dict[str, int] = {}
    for path in sorted((ROOT / "include").rglob("*.h")):
        if path.relative_to(ROOT).as_posix().startswith(SKIP):
            continue
        text = COMMENT.sub("", path.read_text(encoding="utf-8", errors="replace"))
        for m in TYPEDEF.finditer(text):
            end = body_end(text, m.end())
            name = re.match(r"\s*(\w+)\s*;", text[end + 1 :])
            if name:
                out[f"{path.relative_to(ROOT).as_posix()}::{name.group(1)}"] = count_fields(
                    text[m.end() : end]
                )
    return out


def read_baseline() -> dict[str, int]:
    """Recorded widths of the structs that are over the limit."""
    out: dict[str, int] = {}
    if BASELINE.exists():
        for line in BASELINE.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                key, n = line.rsplit(None, 1)
                out[key] = int(n)
    return out


def problems(cur: dict[str, int], base: dict[str, int], limit: int = LIMIT) -> list[str]:
    """Messages for new wide structs and for listed ones that gained fields."""
    out = []
    for key, n in sorted(cur.items()):
        if n <= limit:
            continue
        if key not in base:
            out.append(f"{key}: {n} fields (limit {limit}); split it into sub-structs")
        elif n > base[key]:
            out.append(f"{key}: {n} fields, grew past its recorded {base[key]}")
    return out


def main(argv: list[str]) -> int:
    """Entry point: writes the baseline with --update, else checks the headers; returns the exit code."""
    cur = widths()
    wide = {k: v for k, v in cur.items() if v > LIMIT}
    if "--update" in argv:
        BASELINE.write_text(
            f"# Structs over {LIMIT} fields (tools/check_struct_fields.py): key, fields; entries"
            " may only shrink or go.\n" + "".join(f"{k} {v}\n" for k, v in sorted(wide.items())),
            encoding="utf-8",
        )
        print(f"baseline written: {len(wide)} struct(s)")
        return 0
    found = problems(cur, read_baseline())
    for line in found:
        print(line)
    print(f"structs over {LIMIT} fields: {len(wide)}" if not found else f"{len(found)} problem(s)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
