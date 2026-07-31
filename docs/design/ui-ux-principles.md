# UI/UX Design Principles — project-zero

> Working reference for Phase 22 (web chat UI + CLI polish). Every screenshot captured in
> Phase 22.4 is graded against this checklist. Lives alongside `docs/ai/**` rather than inside
> it (`docs/ai/**` stays the canonical cross-tool memory index; this is a design-QA artifact
> referenced from `docs/ai/project-overview.md`). Last updated: 2026-07-15.

## 1. Best-practice checklist (chat/dev-tool UI)

- **Visual hierarchy**: the thing the user is looking at right now (the streaming reply) must
  be the most visually prominent element on screen — larger contrast against background, more
  whitespace around it, than chrome (topbar, model info, controls).
- **Consistent spacing scale**: pick a small set of spacing values (e.g. 0.25/0.5/0.75/1/1.25rem)
  and reuse them everywhere; never hand-tune one-off pixel values.
- **Accessible contrast (WCAG AA)**: body text ≥ 4.5:1 contrast against its background in both
  themes; dimmed/secondary text ≥ 3:1. Verify both the light and dark palettes, not just one.
- **Responsive**: no fixed pixel widths that break under 1000px; the composer and message list
  must reflow, not overflow horizontally.
- **Clear affordances**: every clickable control looks clickable (cursor, hover state, visible
  border or fill) — no unlabeled icon-only buttons without a `title`/tooltip.
- **Predictable loading/streaming state**: the user must always be able to tell "is it thinking,
  is it streaming, is it done, did it error" without ambiguity — a blinking cursor, a Stop
  button, and a distinct error state each communicate one of these unambiguously.

## 2. Explicit anti-patterns ("don'ts")

- Unlabeled icon-only controls with no `title`/tooltip and no text fallback.
- Inconsistent spacing (mixing arbitrary padding values across sibling components).
- Low-contrast text (light-grey-on-white or dark-grey-on-black body copy).
- Ambiguous loading states (a spinner with no indication of what's loading, or silence with no
  visual feedback at all during a multi-second wait).
- Overflow: horizontal scrollbars on the page body, text/code blocks that break their container.
- Interactive elements smaller than ~32px on touch/click targets (Fitts's law, §4).

## 3. Avoiding the generic "AI-generated" look

The single most concrete, checkable part of this reference — a literal list of tells to avoid,
not a vague aspiration:

- ❌ Purple/violet gradient hero backgrounds or buttons (the single most common tell).
- ❌ Generic centered-hero + 3-feature-card landing layout applied to what should be a
  functional tool screen.
- ❌ Glassmorphism/backdrop-blur used decoratively without an actual layering purpose.
- ❌ Emoji used as functional icons in place of real iconography or clear text labels.
- ❌ Uniform, maximal border-radius on every single element regardless of size/hierarchy
  ("everything is a pill").
- ❌ A single accent color applied indiscriminately to every interactive element with no
  secondary/tertiary hierarchy.
- ✅ Instead: a palette and type system that reflects *this project's own identity* — a
  CPU/systems/terminal heritage — monospace accents for brand/code, a restrained single accent
  color used purposefully (state, not decoration), real borders instead of blur for separation.

## 4. Symmetry, balance, whitespace, grid alignment

- Align elements to a consistent grid/baseline; avoid off-by-a-few-pixels misalignment between
  sibling rows (e.g. topbar left/right groups must share a vertical center line).
- Balance is not always symmetry: a chat UI is legitimately asymmetric (user messages
  right-aligned, assistant left-aligned) — this is intentional and should be preserved, not
  "fixed" into visual symmetry that would hurt scanability.
- Generous whitespace around the primary content column; don't let message bubbles stretch
  edge-to-edge on wide viewports (cap line length for readability, ~60-75ch equivalent).

## 5. Gestalt / psychological principles (why they matter here)

- **Proximity**: group a message's role label with its content tightly, and space distinct
  messages apart — proximity alone communicates "these belong together" without borders.
- **Similarity**: user vs. assistant messages should be visually distinguishable by a consistent
  signal (background fill vs. border), so the eye classifies them instantly while scrolling.
- **Figure-ground**: the composer (input) must read as "in front of"/separate from the message
  history behind it — a border-top + solid background achieves this without heavy shadows.
- **Contrast/visual hierarchy**: the accent color should appear only on things that matter
  (online indicator, primary action, active state) — overuse flattens hierarchy into noise.
- **Fitts's law**: primary actions (Send/Stop) should be large enough and close enough to the
  input to acquire quickly; don't bury them behind a menu.
- **Von Restorff effect (isolation)**: an error state should look distinctly different from
  normal content (color + icon, not just color) so it's never missed while scanning.

## 6. CLI-specific notes (Phase 22.3)

The same anti-"generic AI" instinct applies to terminal output: avoid rainbow/gratuitous ANSI
color on every line (reserve color for status: green=success/online, dim=secondary/live-stats,
red=error/stop); respect `NO_COLOR` and non-TTY output unconditionally; a progress indicator
should degrade to a single clean line off-TTY, never leak raw escape codes into piped output.
