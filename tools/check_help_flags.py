#!/usr/bin/env python3
"""Fails when a command-line flag that parse_args() accepts is missing from the --help text.

The flags come from `strcmp(argv[i], "--flag")` in src/cli/args.c and the flag groups (args_flags_*.c), the help text from
src/cli/args_usage.c. A flag nobody can discover is a flag nobody uses (and DoD #6 wants every new
setting visible), so adding a flag without its help line fails CI.

usage: tools/check_help_flags.py
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARGS_FILES = sorted((ROOT / "src" / "cli").glob("args*.c"))
USAGE = ROOT / "src" / "cli" / "args_usage.c"

PARSED = re.compile(r'strcmp\(argv\[i\],\s*"(--[a-z0-9-]+)"\)')
MENTIONED = re.compile(r"(--[a-z0-9-]+)")


def parsed_flags(text: str) -> set[str]:
    """Long flags that the parser matches."""
    return set(PARSED.findall(text))


def documented_flags(text: str) -> set[str]:
    """Long flags that appear anywhere in the help text."""
    return set(MENTIONED.findall(text))


def undocumented(parsed: set[str], documented: set[str]) -> list[str]:
    """Sorted flags that are parsed but never mentioned in the help text."""
    return sorted(parsed - documented)


def main() -> int:
    """Entry point; returns 1 when a parsed flag has no help text."""
    parsed = set()
    for path in ARGS_FILES:
        if path.name != "args_usage.c":
            parsed |= parsed_flags(path.read_text(encoding="utf-8"))
    if not parsed:
        print("no flags found in args.c (did the parser format change?)")
        return 2
    missing = undocumented(parsed, documented_flags(USAGE.read_text(encoding="utf-8")))
    for flag in missing:
        print(f"{flag} is parsed by parse_args() but not described in --help")
    print(f"help flags: {len(parsed)} parsed, {len(missing)} undocumented")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
