# Changelog Guidance — project-zero

> Canonical rules for `CHANGELOG.md` (the user-facing version history). Read before any change a
> user could notice, and before cutting a release. Last updated: 2026-10-21.
>
> Sources: [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/) (format, change types,
> Unreleased, yanked releases), [Common Changelog](https://common-changelog.org/) (entry wording,
> breaking-change marking, curation), [Semantic Versioning 2.0](https://semver.org/) (version
> numbers). Where they differ, the choice made for this repo is stated below.

## 1. Principles
- **Written for people who run Project Zero, not for developers.** Say what changed for them and
  which flag, API field or UI control it touches. Never paste `git log` output or a list of PRs;
  commits are raw input to review, merge and rewrite (Keep a Changelog: "commit log diffs" are a
  bad practice).
- **Complete for what matters.** "A changelog which only mentions some of the changes can be as
  dangerous as not having a changelog." At minimum, every deprecation, removal and breaking change
  is listed.
- **Every version has an entry; newest first; dates are ISO 8601 (`YYYY-MM-DD`); versions and
  sections are linkable.** The file says it follows Keep a Changelog and SemVer.

## 2. When to update
- **In the same commit as the change** (engineering-rules DoD #5). The line goes under
  `## [Unreleased]`, which always exists at the top.
- What needs an entry — anything a user can observe:
  new/changed/removed CLI flags, env vars, API routes or fields, web UI controls, defaults,
  supported models/quant formats/architectures, output or behaviour changes, notable performance
  or memory changes (with the number and the host), bug fixes users could have hit, security fixes,
  release-asset or binary-name changes, build requirements (compiler, glibc floor).
- What does **not**: internal refactors with no visible effect, tests, CI plumbing, benchmark logs,
  doc formatting, AI-adapter/process docs, dev-only tooling. (Keep: new user documentation, runtime
  environment support changes.)
- Fixing a bug introduced and fixed within the same `[Unreleased]` window → no entry (the two
  changes cancel out); fold any user-visible remainder into the original entry instead.
- Past entries may be corrected (Keep a Changelog FAQ): add a missed breaking change or release,
  fix a wrong statement. Do not silently delete a released entry.

## 3. Format (enforced by `python3 tools/changelog.py check`)
```markdown
# Changelog
(one paragraph: follows Keep a Changelog 1.1.0 and SemVer)

## [Unreleased]

### Changed
- **Breaking:** Rename the binary to `pz` ...

## [0.2.0] - 2026-11-01

### Added
- Add `--kv-type i4` ...

[Unreleased]: https://github.com/<owner>/project-zero/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/<owner>/project-zero/compare/v0.1.0...v0.2.0
```
- Release heading: `## [X.Y.Z] - YYYY-MM-DD` — SemVer without the `v` (tags are `vX.Y.Z`); a
  pulled release keeps its entry with ` [YANKED]` appended plus a one-sentence emphasised notice
  saying why and what to use instead.
- Sections, only these, in this order, empty ones omitted:
  `Changed`, `Deprecated`, `Removed`, `Added`, `Fixed`, `Security`
  (Keep a Changelog's six types; ordered breaking-first as Common Changelog does).
- Each section is a plain bullet list. A release may start with **one** emphasised notice sentence
  (upgrade pointer, first release, yanked status); it never replaces the list.
- Every version heading has a reference link at the bottom (compare view; tag page for the first).

## 4. Writing an entry
- **One line, imperative mood, self-describing without its heading** (Common Changelog):
  "Add `--kv-type i4` for int4 KV cache storage", "Fix generation not stopping at end-of-turn
  tokens", not "int4 KV" or "Fixed a bug".
- **Name the interface**: the flag, env var, API route/field or UI location in backticks. Give the
  default when it matters.
- **Breaking changes** start with `**Breaking:**` and say what to do:
  "**Breaking:** Rename the binary to `pz`; scripts calling `adaptive_ai_engine` keep working
  for one release through an alias". Breaking entries come first in their section. While on 0.x
  a breaking change bumps MINOR; at ≥ 1.0 it bumps MAJOR.
- **Deprecations** name the replacement and the release that removes it. A removal in a later
  release gets a `Removed` entry pointing back to the deprecation.
- **Performance** entries quote the measured change and the conditions
  ("decode +18 % on DeepSeek-V2-Lite Q4_K_S, 4 threads, Xeon AVX-512") — never an unmeasured claim.
- **Security** entries state impact and affected versions; link the advisory (GHSA/CVE) when one
  exists. Do not describe an unpatched vulnerability before the fixed release is out
  (`SECURITY.md`).
- **References** (optional here): link the PR or issue in parentheses after the text when one
  exists — `([#32](https://github.com/<owner>/project-zero/issues/32))`. Most work lands as direct
  commits, so a reference is not required. No author names (single-maintainer project) and **no AI
  attribution**.
- Merge related commits into one entry; keep wording consistent with the CLI/API names; detail
  belongs in commits, docs and linked issues, not in the changelog.

## 5. Releasing (full procedure: `docs/RELEASING.md`)
1. Pick the version by SemVer from what `[Unreleased]` contains (`Removed`/breaking → MINOR on
   0.x, MAJOR from 1.0; `Added`/`Deprecated` → MINOR; only `Fixed`/`Security` → PATCH).
2. Re-read `[Unreleased]` as a user: merge duplicates, drop noise, order breaking first, check
   flags/fields still exist under those names.
3. Rename it to `## [X.Y.Z] - <today>`, add a fresh empty `## [Unreleased]` above it, update the
   reference links. `tools/changelog.py check` must pass; preview with
   `tools/changelog.py notes vX.Y.Z`.
4. Commit the changelog **before** tagging; tag that commit `vX.Y.Z`. The release workflow refuses
   a tag without a changelog section and publishes that section as the GitHub release notes, so the
   release page, the update banner link and the web UI's "Release notes" link show the same text.
5. Prereleases (`X.Y.Z-rc.N`): either give them their own entry and point the final release's
   notice at it, or write the final entry as if the prereleases never existed. Internal-only test
   builds get no entry.

## 6. Checklist (before committing a user-facing change)
- [ ] Line added under `[Unreleased]` in the right section, imperative, one line, interface named.
- [ ] Breaking? `**Breaking:**` prefix + what to do. Deprecation? replacement + removal release.
- [ ] Numbers for performance claims; advisory link for security fixes.
- [ ] `python3 tools/changelog.py check --require-unreleased` passes.
