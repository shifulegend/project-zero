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

# Keep a Changelog types, ordered breaking-first (docs/ai/changelog-guidance.md §3).
SECTIONS = ("Changed", "Deprecated", "Removed", "Added", "Fixed", "Security")
SEMVER = r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?"
RELEASE_RE = re.compile(r"^## \[(" + SEMVER + r")\] - (\d{4}-\d{2}-\d{2})( \[YANKED\])?$")
LINK_RE = re.compile(r"^\[([^\]]+)\]: (\S+)$")


def parse(text):
    """-> (entries [(name, date|None, line_no, body_lines)], links {name: url}, errors)."""
    entries, links, errors = [], {}, []
    cur = None
    for n, line in enumerate(text.splitlines(), 1):
        m = LINK_RE.match(line)
        if m:
            links[m.group(1)] = m.group(2)
            continue
        if line.startswith("## "):
            if line == "## [Unreleased]":
                cur = ["Unreleased", None, n, []]
            else:
                r = RELEASE_RE.match(line)
                if not r:
                    errors.append(f"line {n}: bad version heading {line!r} "
                                  "(want '## [X.Y.Z] - YYYY-MM-DD[ [YANKED]]' or '## [Unreleased]')")
                    cur = None
                    continue
                cur = [r.group(1), r.group(5), n, []]
            entries.append(cur)
            continue
        if line.startswith("### ") and cur is not None:
            name = line[4:].strip()
            if name not in SECTIONS:
                errors.append(f"line {n}: subsection {name!r} not one of {', '.join(SECTIONS)}")
            else:
                done = [l[4:].strip() for l in cur[3] if l.startswith("### ")]
                if name in done:
                    errors.append(f"line {n}: subsection {name!r} repeated in {cur[0]}")
                elif any(SECTIONS.index(d) > SECTIONS.index(name) for d in done if d in SECTIONS):
                    errors.append(f"line {n}: subsection {name!r} out of order "
                                  f"(order: {', '.join(SECTIONS)})")
        if cur is not None:
            cur[3].append(line)
    return entries, links, errors


def _key(v):
    core = v.split("-", 1)[0]
    pre = v.split("-", 1)[1] if "-" in v else None
    # A pre-release sorts below its release.
    return tuple(int(x) for x in core.split(".")) + ((0, pre) if pre else (1, ""),)


def check(text, require_unreleased=False):
    entries, links, errors = parse(text)
    names = [e[0] for e in entries]
    if names.count("Unreleased") > 1:
        errors.append("more than one [Unreleased] section")
    if "Unreleased" in names and names[0] != "Unreleased":
        errors.append("[Unreleased] must be the first section")
    if require_unreleased and "Unreleased" not in names:
        errors.append("no [Unreleased] section")
    rel = [e for e in entries if e[0] != "Unreleased"]
    seen = set()
    for e in rel:
        if e[0] in seen:
            errors.append(f"line {e[2]}: duplicate version {e[0]}")
        seen.add(e[0])
    for a, b in zip(rel, rel[1:]):
        if _key(a[0]) <= _key(b[0]):
            errors.append(f"line {b[2]}: {b[0]} listed after {a[0]} (newest first)")
        if a[1] < b[1]:
            errors.append(f"line {b[2]}: date {b[1]} is later than {a[0]}'s {a[1]}")
    for e in entries:
        if e[0] not in links:
            errors.append(f"line {e[2]}: no link reference '[{e[0]}]: <url>' at the bottom")
    return errors


def notes(text, version):
    v = version[1:] if version[:1] in "vV" and version[1:2].isdigit() else version
    entries, links, _ = parse(text)
    for name, _date, _n, body in entries:
        if name.lower() == v.lower():
            out = "\n".join(body).strip()
            if not out:
                return None
            if name in links:
                out += f"\n\n**Full changes:** {links[name]}"
            return out + "\n"
    return None


def main(argv=None):
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
        print(f"{a.file}: no non-empty section for {a.version!r} — add release notes first",
              file=sys.stderr)
        return 1
    sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
