# Commit & Log Guidance — project-zero

> How to make small, traceable commits. Last updated: 2026-06-07.

## Granularity
- One coherent, reviewable sub-step per commit. Split test fixes, build fixes, and docs when
  they are logically distinct.
- Commit after verification (build/test/golden output green). If git isn't available, state
  the exact commit message that should be created.

## Message format
- Conventional-commit prefix: `feat|fix|perf|docs|test|build|refactor|chore(scope): summary`.
- Body: **what** changed and **why** (root cause / rationale), plus what was **verified**.
- Reference affected modules; link the mistake/decision entry if the commit closes one.
- Keep model identifiers / tool-internal IDs out of commit messages.

## Examples (from this repo)
- `fix(ci): memset TransformerWeights before weights_alloc_pointers in blackbox/audit tests`
- `build(make): link sanitizer runtime + add -march=native/-mavx512vl to debug build`
- `fix(run_state): deterministic OOM guard for absurd allocations (cross-platform)`
- `docs(ai): seed canonical project memory + tool adapters`

## Checkpoint discipline
- After each meaningful sub-step: propose the commit message and update the relevant
  `docs/ai/**` entry (`mistakes`/`decision-log`/`change-trace`) in the same or adjacent commit.
- Never bundle an unrelated refactor into a fix.
