#!/usr/bin/env python3
"""Generate the Antigravity and Copilot scoped-rule files from .claude/rules/*.md.

One source, three tools (docs/ai/tool-sync-policy.md, 2026-11-11):
  .claude/rules/<scope>.md                      source; YAML `paths:` list + "# Claude rule: <title>"
  .agents/rules/<scope>.md                      generated (Antigravity)
  .github/instructions/<scope>.instructions.md  generated (Copilot; `applyTo` from `paths:`)

Usage:
  tools/sync_ai_adapters.py           rewrite the generated files
  tools/sync_ai_adapters.py --check   exit 1 if any generated file is stale (CI)

Before 2026-11-11 the three copies were edited by hand and drifted (bullets present in one
tool's copy only). Edit the .claude/rules file, then run this script.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / ".claude" / "rules"
AGENTS_DIR = ROOT / ".agents" / "rules"
COPILOT_DIR = ROOT / ".github" / "instructions"


def note(src_rel):
    return (f"<!-- Generated from {src_rel} by tools/sync_ai_adapters.py; "
            f"edit that file and re-run the script (CI checks this copy). -->")


def parse(path):
    """Return (paths, title, body) of a .claude/rules file."""
    text = path.read_text(encoding="utf-8")
    paths = []
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if m:
        in_paths = False
        for line in m.group(1).splitlines():
            if re.match(r"^paths:\s*$", line):
                in_paths = True
                continue
            item = re.match(r'^\s+-\s+"?([^"]+?)"?\s*$', line)
            if in_paths and item:
                paths.append(item.group(1))
            elif line and not line.startswith(" "):
                in_paths = False
        text = text[m.end():]
    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines or not lines[0].startswith("# Claude rule:"):
        sys.exit(f"{path}: expected a '# Claude rule: <title>' heading after the front matter")
    title = lines[0][len("# Claude rule:"):].strip()
    body = "\n".join(lines[1:]).strip("\n") + "\n"
    return paths, title, body


def render(path):
    paths, title, body = parse(path)
    if not paths:
        sys.exit(f"{path}: needs a non-empty `paths:` list (Copilot applyTo is built from it)")
    src_rel = path.relative_to(ROOT).as_posix()
    scope = path.stem
    agents = f"# Antigravity scoped rules: {title}\n\n{note(src_rel)}\n\n{body}"
    copilot = ("---\n"
               f'applyTo: "{",".join(paths)}"\n'
               "---\n"
               f"# Copilot scoped rules: {title}\n\n{note(src_rel)}\n\n{body}")
    return {
        AGENTS_DIR / f"{scope}.md": agents,
        COPILOT_DIR / f"{scope}.instructions.md": copilot,
    }


def main(argv):
    check = "--check" in argv[1:]
    sources = sorted(SRC_DIR.glob("*.md"))
    if not sources:
        sys.exit(f"no source rules in {SRC_DIR}")
    stale = []
    for src in sources:
        for out, content in render(src).items():
            current = out.read_text(encoding="utf-8") if out.exists() else None
            if current == content:
                continue
            stale.append(out.relative_to(ROOT).as_posix())
            if not check:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(content, encoding="utf-8")
    if check and stale:
        print("Stale generated adapter files (edit .claude/rules/*.md, then run "
              "tools/sync_ai_adapters.py):")
        for s in stale:
            print(f"  {s}")
        return 1
    if not check:
        print(f"updated {len(stale)} file(s)" if stale else "all adapter files up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
