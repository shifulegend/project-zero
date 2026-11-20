"""Duplicated code must not increase (a ratchet, like the complexity and size checks).

For each scope (C/C++ in src/, TypeScript and Svelte in webui/src/) the check normalises every
file (comments, blank lines and brace-only lines dropped, whitespace trimmed), slides a window of
WINDOW consecutive lines over it and counts the lines covered by a window that also occurs
elsewhere. The share of such lines is compared with tools/duplication_baseline.txt (percent with
two decimals); a higher share fails, `--update` rewrites the baseline when it went down.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tools" / "duplication_baseline.txt"
WINDOW = 6
SCOPES: dict[str, tuple[str, tuple[str, ...]]] = {
    "src": ("src", (".c", ".cpp")),
    "webui": ("webui/src", (".ts", ".svelte")),
}
SKIP = {"webui_bundle_generated.c"}
NOISE = re.compile(r"^[{}();,]*$|^(//|/\*|\*)")


def normalised(text: str) -> list[str]:
    """The code lines of a file: no blanks, comment lines or lone braces, whitespace collapsed."""
    lines: list[str] = []
    in_block = False
    for raw in text.splitlines():
        line = raw.strip()
        if in_block:
            in_block = "*/" not in line
            continue
        if line.startswith("/*") and "*/" not in line:
            in_block = True
            continue
        if not line or NOISE.match(line):
            continue
        lines.append(re.sub(r"\s+", " ", line))
    return lines


def duplicated_share(files: dict[str, list[str]], window: int = WINDOW) -> tuple[int, int]:
    """(lines covered by a repeated window, total code lines) over the given files."""
    seen: Counter[tuple[str, ...]] = Counter()
    for lines in files.values():
        for i in range(len(lines) - window + 1):
            seen[tuple(lines[i : i + window])] += 1
    covered = 0
    total = 0
    for lines in files.values():
        total += len(lines)
        marks = [False] * len(lines)
        for i in range(len(lines) - window + 1):
            if seen[tuple(lines[i : i + window])] > 1:
                for k in range(i, i + window):
                    marks[k] = True
        covered += sum(marks)
    return covered, total


def measure() -> dict[str, float]:
    """The duplicated share in percent for each scope."""
    result: dict[str, float] = {}
    for name, (folder, suffixes) in SCOPES.items():
        files = {
            str(path): normalised(path.read_text(encoding="utf-8", errors="replace"))
            for path in sorted((ROOT / folder).rglob("*"))
            if path.suffix in suffixes and path.name not in SKIP
        }
        covered, total = duplicated_share(files)
        result[name] = round(100.0 * covered / total, 2) if total else 0.0
    return result


def read_baseline() -> dict[str, float]:
    """The recorded shares, name -> percent."""
    out: dict[str, float] = {}
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            name, value = line.split()
            out[name] = float(value)
    return out


def main(argv: list[str]) -> int:
    """Compare the measured shares with the baseline (or rewrite it with --update)."""
    now = measure()
    if "--update" in argv:
        header = "# Share of duplicated code lines in percent (tools/check_duplication.py); may only fall.\n"
        BASELINE.write_text(
            header + "".join(f"{k} {v:.2f}\n" for k, v in now.items()), encoding="utf-8"
        )
        print("baseline written:", now)
        return 0
    base = read_baseline()
    problems = 0
    for name, value in now.items():
        if value > base.get(name, 0.0) + 1e-9:
            print(
                f"duplication in {name}: {value:.2f}% is above the recorded {base.get(name, 0.0):.2f}%"
            )
            problems += 1
        elif value < base.get(name, 0.0) - 0.5:
            print(
                f"duplication in {name} fell to {value:.2f}% (recorded {base[name]:.2f}%): run --update"
            )
            problems += 1
    print("duplication:", ", ".join(f"{k} {v:.2f}%" for k, v in now.items()))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
