#ifndef TN_CLI_BANNER_H
#define TN_CLI_BANNER_H

/*
 * Phase 22.5 — animated ASCII-art startup banner ("PROJECT ZERO"), the kind
 * of splash seen in Claude Code and other modern CLI tools. TTY-gated (no
 * escape codes ever leak into piped/redirected output) and shown only for
 * long-running, human-watched invocations (REPL, --server) — not for
 * one-shot --prompt runs, matching how Claude Code itself suppresses its
 * banner in non-interactive/scripted ("-p") mode.
 */

/* Prints the banner with a line-by-line reveal, bottom row first, each new
 * row appearing above the previous (a "sliding up into place" effect),
 * followed by a brief bounded dim/bold "shimmer" pulse once fully revealed
 * (color_enabled only — with color off there is nothing to pulse). No-op
 * entirely when is_tty is false. color_enabled controls whether the glyphs
 * are tinted with the CLI's accent color (see cli/color.h). */
void tn_banner_print(int is_tty, int color_enabled);

#endif /* TN_CLI_BANNER_H */
