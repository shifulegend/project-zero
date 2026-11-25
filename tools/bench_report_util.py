"""Helpers shared by the benchmark report scripts: CSV readers and number formatting."""

import csv
import io
import os
import re

Row = dict[str, str]  # one CSV row, header -> text


def read_csv(path: str) -> list[Row]:
    """Read CSV, tolerating continuation lines (bash printf newline bug)."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    # Fix: join lines that don't start with a date prefix or 'date' header
    fixed = re.sub(r"\n(?!20[0-9]|date)", ",", raw)
    rows: list[Row] = []
    reader = csv.DictReader(io.StringIO(fixed))
    for row in reader:
        # Clean up any extra commas creating phantom fields
        rows.append({k: v.strip() for k, v in row.items() if k is not None})
    return rows


def read_llama_csv(path: str) -> list[Row]:
    """Read llama-bench CSV (clean format, no fixing needed)."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_file(path: str) -> str:
    """Contents of a file, or "(not found)" when it does not exist."""
    if not os.path.exists(path):
        return "(not found)"
    with open(path, encoding="utf-8") as f:
        return f.read()


def safe_float(v: str | float | None, default: float = 0.0) -> float:
    """`v` as a float, or `default` when it is missing or not a number."""
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def fmt_tps(v: str | float | None) -> str:
    """A tok/s value formatted for the report, "N/A" when missing or zero."""
    f = safe_float(v)
    if f == 0:
        return "N/A"
    return f"{f:.3f}"


def speedup(our: str | float | None, llama: str | float | None) -> str:
    """Speed-up of the first value over the second as "N.N×", "N/A" when either is missing."""
    o = safe_float(our)
    l = safe_float(llama)
    if o == 0 or l == 0:
        return "N/A"
    return f"{l / o:.1f}×"
