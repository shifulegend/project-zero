#!/usr/bin/env python3
"""Fails when line or branch coverage falls below the floor in tools/coverage_floor.txt.

Reads an lcov tracefile (build/coverage/coverage.info by default): LF/LH are lines found/hit,
BRF/BRH branches found/hit. Usage: check_coverage.py [coverage.info]
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FLOOR = ROOT / "tools" / "coverage_floor.txt"
STALE_MARGIN = (
    5.0  # points of headroom before the floor must be raised (CPU-dependent kernel paths)
)
DEFAULT_INFO = ROOT / "build" / "coverage" / "coverage.info"


def totals(text: str) -> dict[str, tuple[int, int]]:
    """Sums the per-file records of a tracefile into {'lines': (hit, found), 'branches': ...}."""
    sums = {"LF": 0, "LH": 0, "BRF": 0, "BRH": 0}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key in sums and value.strip().isdigit():
            sums[key] += int(value)
    return {"lines": (sums["LH"], sums["LF"]), "branches": (sums["BRH"], sums["BRF"])}


def percent(hit_found: tuple[int, int]) -> float:
    """Percentage for a (hit, found) pair; 0 when nothing was found."""
    hit, found = hit_found
    return 100.0 * hit / found if found else 0.0


def read_floor(text: str) -> dict[str, float]:
    """Parses the floor file into {metric: minimum percent}, ignoring comments."""
    floor = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            name, value = line.split()
            floor[name] = float(value)
    return floor


def check(info_text: str, floor: dict[str, float]) -> list[str]:
    """Returns a list of failure messages (empty when every metric meets its floor)."""
    got = totals(info_text)
    failures = []
    for name, minimum in floor.items():
        if name not in got:
            failures.append(f"unknown metric in the floor file: {name}")
            continue
        found = got[name][1]
        pct = percent(got[name])
        if found == 0:
            failures.append(f"{name}: no data in the tracefile")
        elif pct < minimum:
            failures.append(f"{name} coverage {pct:.1f}% is below the floor {minimum:.1f}%")
        elif pct > minimum + STALE_MARGIN:
            failures.append(
                f"{name} coverage {pct:.1f}% is more than {STALE_MARGIN:.0f} points above the floor"
                f" {minimum:.1f}%: raise it in tools/coverage_floor.txt (the floor only goes up)"
            )
    return failures


def main(argv: list[str]) -> int:
    """Entry point: compares the tracefile's coverage with the floor and returns the exit code."""
    info = pathlib.Path(argv[1]) if len(argv) > 1 else DEFAULT_INFO
    if not info.exists():
        print(f"coverage tracefile not found: {info} (run `make coverage` first)")
        return 2
    text = info.read_text(encoding="utf-8")
    failures = check(text, read_floor(FLOOR.read_text(encoding="utf-8")))
    got = totals(text)
    print(f"coverage: lines {percent(got['lines']):.1f}%, branches {percent(got['branches']):.1f}%")
    for f in failures:
        print(f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
