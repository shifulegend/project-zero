#!/usr/bin/env python3
"""
changelog.py — lint CHANGELOG.md (Keep a Changelog 1.1.0 + SemVer) and extract
one version's section as release notes.

  python3 tools/changelog.py check [--file CHANGELOG.md] [--require-unreleased]
  python3 tools/changelog.py notes <version|vX.Y.Z|Unreleased> [--file CHANGELOG.md]

Rules: docs/ai/changelog-guidance.md.
check: exactly one "## [Unreleased]" and it comes first; releases are
"## [X.Y.Z] - YYYY-MM-DD" (optionally " [YANKED]"), newest first, no
duplicates; only the six standard subsections, each at most once, in the order
Changed, Deprecated, Removed, Added, Fixed, Security; every version heading has
a link reference at the bottom.
notes: prints the body of that section (without its heading) plus a link to
the full diff; exits 1 if the section is missing or empty, so a tag cannot be
released without changelog notes.
"""

import argparse
import re
import sys
from dataclasses import dataclass, field

# Keep a Changelog types, ordered breaking-first (docs/ai/changelog-guidance.md §3).
SECTIONS = ("Changed", "Deprecated", "Removed", "Added", "Fixed", "Security")
SEMVER = r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?"
RELEASE_RE = re.compile(r"^## \[(" + SEMVER + r")\] - (\d{4}-\d{2}-\d{2})( \[YANKED\])?$")
LINK_RE = re.compile(r"^\[([^\]]+)\]: (\S+)$")


@dataclass
class Entry:
    """One `## [version]` section of the changelog."""

    name: str
    date: str | None
    line_no: int
    body: list[str] = field(default_factory=list)


def parse(text: str) -> tuple[list[Entry], dict[str, str], list[str]]:
    """-> (entries, links {name: url}, errors)."""
    entries: list[Entry] = []
    links: dict[str, str] = {}
    errors: list[str] = []
    cur: Entry | None = None
    for n, line in enumerate(text.splitlines(), 1):
        m = LINK_RE.match(line)
        if m:
            links[m.group(1)] = m.group(2)
            continue
        if line.startswith("## "):
            if line == "## [Unreleased]":
                cur = Entry("Unreleased", None, n)
            else:
                r = RELEASE_RE.match(line)
                if not r:
                    errors.append(
                        f"line {n}: bad version heading {line!r} "
                        "(want '## [X.Y.Z] - YYYY-MM-DD[ [YANKED]]' or '## [Unreleased]')"
                    )
                    cur = None
                    continue
                cur = Entry(r.group(1), r.group(5), n)
            entries.append(cur)
            continue
        if line.startswith("### ") and cur is not None:
            name = line[4:].strip()
            if name not in SECTIONS:
                errors.append(f"line {n}: subsection {name!r} not one of {', '.join(SECTIONS)}")
            else:
                done = [l[4:].strip() for l in cur.body if l.startswith("### ")]
                if name in set(done):
                    errors.append(f"line {n}: subsection {name!r} repeated in {cur.name}")
                elif any(SECTIONS.index(d) > SECTIONS.index(name) for d in done if d in SECTIONS):
                    errors.append(
                        f"line {n}: subsection {name!r} out of order (order: {', '.join(SECTIONS)})"
                    )
        if cur is not None:
            cur.body.append(line)
    return entries, links, errors


def _key(v: str) -> tuple[int | tuple[int, str], ...]:
    core = v.split("-", 1)[0]
    pre = v.split("-", 1)[1] if "-" in v else None
    # A pre-release sorts below its release.
    parts: list[int | tuple[int, str]] = [int(x) for x in core.split(".")]
    parts.append((0, pre) if pre else (1, ""))
    return tuple(parts)


def check(text: str, require_unreleased: bool = False) -> list[str]:
    """Problems found in a CHANGELOG (structure, [Unreleased] placement, links); `require_unreleased` also demands a non-empty [Unreleased] section."""
    entries, links, errors = parse(text)
    names = [e.name for e in entries]
    if names.count("Unreleased") > 1:
        errors.append("more than one [Unreleased] section")
    if "Unreleased" in names and names[0] != "Unreleased":
        errors.append("[Unreleased] must be the first section")
    if require_unreleased and "Unreleased" not in names:
        errors.append("no [Unreleased] section")
    rel = [e for e in entries if e.name != "Unreleased"]
    seen: set[str] = set()
    for e in rel:
        if e.name in seen:
            errors.append(f"line {e.line_no}: duplicate version {e.name}")
        seen.add(e.name)
    for a, b in zip(rel, rel[1:], strict=False):
        if _key(a.name) <= _key(b.name):
            errors.append(f"line {b.line_no}: {b.name} listed after {a.name} (newest first)")
        if (a.date or "") < (b.date or ""):
            errors.append(f"line {b.line_no}: date {b.date} is later than {a.name}'s {a.date}")
    for e in entries:
        if e.name not in links:
            errors.append(f"line {e.line_no}: no link reference '[{e.name}]: <url>' at the bottom")
    return errors


def notes(text: str, version: str) -> str | None:
    """Release notes for `version` (its section plus the full-changes link), or None when the section is absent or empty."""
    v = version[1:] if version[:1] in "vV" and version[1:2].isdigit() else version
    entries, links, _ = parse(text)
    for entry in entries:
        name = entry.name
        if name.lower() == v.lower():
            out = "\n".join(entry.body).strip()
            if not out:
                return None
            if name in links:
                out += f"\n\n**Full changes:** {links[name]}"
            return out + "\n"
    return None


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: the check and notes subcommands."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--file", default="CHANGELOG.md")
    c.add_argument("--require-unreleased", action="store_true")
    nt = sub.add_parser("notes")
    nt.add_argument("version")
    nt.add_argument("--file", default="CHANGELOG.md")
    a = ap.parse_args(argv)
    with open(a.file, encoding="utf-8") as f:
        text = f.read()
    if a.cmd == "check":
        errs = check(text, a.require_unreleased)
        for e in errs:
            print(f"{a.file}: {e}", file=sys.stderr)
        if not errs:
            print(f"{a.file}: ok")
        return 1 if errs else 0
    out = notes(text, a.version)
    if out is None:
        print(
            f"{a.file}: no non-empty section for {a.version!r} — add release notes first",
            file=sys.stderr,
        )
        return 1
    sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
