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


def note(src_rel: str) -> str:
    """The "generated from" header placed on every adapter file."""
    return (
        f"<!-- Generated from {src_rel} by tools/sync_ai_adapters.py; "
        "edit that file and re-run the script (CI checks this copy). -->"
    )


def parse(path: pathlib.Path) -> tuple[list[str], str, str]:
    """Return (paths, title, body) of a .claude/rules file."""
    text = path.read_text(encoding="utf-8")
    paths: list[str] = []
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
        text = text[m.end() :]
    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines or not lines[0].startswith("# Claude rule:"):
        sys.exit(f"{path}: expected a '# Claude rule: <title>' heading after the front matter")
    title = lines[0][len("# Claude rule:") :].strip()
    body = "\n".join(lines[1:]).strip("\n") + "\n"
    return paths, title, body


def render(path: pathlib.Path) -> dict[pathlib.Path, str]:
    """Renders one source rule file into its adapter text (Copilot or Antigravity form)."""
    paths, title, body = parse(path)
    if not paths:
        sys.exit(f"{path}: needs a non-empty `paths:` list (Copilot applyTo is built from it)")
    src_rel = path.relative_to(ROOT).as_posix()
    scope = path.stem
    agents = f"# Antigravity scoped rules: {title}\n\n{note(src_rel)}\n\n{body}"
    copilot = (
        "---\n"
        f'applyTo: "{",".join(paths)}"\n'
        "---\n"
        f"# Copilot scoped rules: {title}\n\n{note(src_rel)}\n\n{body}"
    )
    return {
        AGENTS_DIR / f"{scope}.md": agents,
        COPILOT_DIR / f"{scope}.instructions.md": copilot,
    }


# Durable rules every root entrypoint must carry (a one-line mention each is enough; the full
# text lives in docs/ai/engineering-rules.md). Added 2026-11-19 after GEMINI.md and the Copilot
# file had silently lost several of them.
ENTRYPOINTS = ["CLAUDE.md", "AGENTS.md", "gemini/GEMINI.md", ".github/copilot-instructions.md"]
REQUIRED_RULES = [
    ("definition of done (gcc and clang)", r"gcc and clang"),
    ("installer + pz config schema (DoD #6)", r"DoD #6"),
    ("README review (DoD #7)", r"DoD #7"),
    ("immutable published releases", r"Published releases are immutable"),
    ("no AI attribution", r"[Aa]ttribution"),
    ("bug-fix policy", r"Any bug found gets fixed"),
]


def persona_drift() -> list[str]:
    """The Copilot persona must carry the same instructions as the Claude one (front matter differs)."""
    out: list[str] = []
    src_dir = ROOT / ".claude" / "agents"
    for src in sorted(src_dir.glob("*.md")) if src_dir.exists() else []:
        dst = ROOT / ".github" / "agents" / f"{src.stem}.agent.md"

        def body(t: str) -> str:
            """The text of a persona file without its front matter."""
            return re.sub(r"^---\n.*?\n---\n", "", t, count=1, flags=re.S)

        if not dst.exists():
            out.append(
                f"{dst.relative_to(ROOT).as_posix()}: missing copy of"
                f" {src.relative_to(ROOT).as_posix()}"
            )
        elif body(src.read_text(encoding="utf-8")) != body(dst.read_text(encoding="utf-8")):
            out.append(
                f"{dst.relative_to(ROOT).as_posix()}: body differs from"
                f" {src.relative_to(ROOT).as_posix()}"
            )
    return out


def missing_rules() -> list[str]:
    """Entry points (CLAUDE.md, AGENTS.md, ...) that are missing a durable rule."""
    out: list[str] = []
    for ep in ENTRYPOINTS:
        path = ROOT / ep
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        for label, pat in REQUIRED_RULES:
            if not re.search(pat, text):
                out.append(f"{ep}: missing durable rule '{label}'")
    return out


def main(argv: list[str]) -> int:
    """Entry point: regenerates the adapter files, or with --check verifies they are current; returns the exit code."""
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
    gaps = (missing_rules() + persona_drift()) if check else []
    if check and gaps:
        print("Root adapter files lost a durable rule (copy it from CLAUDE.md / AGENTS.md):")
        for g in gaps:
            print(f"  {g}")
        if not stale:
            return 1
    if check and stale:
        print(
            "Stale generated adapter files (edit .claude/rules/*.md, then run "
            "tools/sync_ai_adapters.py):"
        )
        for s in stale:
            print(f"  {s}")
        return 1
    if not check:
        print(f"updated {len(stale)} file(s)" if stale else "all adapter files up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
