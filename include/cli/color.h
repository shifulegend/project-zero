#ifndef TN_CLI_COLOR_H
#define TN_CLI_COLOR_H

/*
 * Phase 22.3 — CLI color support.
 * Pure/testable resolution logic; real call sites pass isatty(fileno(stdout))
 * and getenv("NO_COLOR") so this file has no TTY/env dependency itself.
 */

typedef enum { TN_COLOR_AUTO = 0, TN_COLOR_ALWAYS, TN_COLOR_NEVER } TnColorMode;

/* Parses "auto"/"always"/"never" (case-insensitive). Returns TN_COLOR_AUTO
 * for NULL or any unrecognized value — callers that need strict validation
 * (e.g. CLI arg parsing) should reject unrecognized strings themselves
 * before falling back to this. */
TnColorMode tn_color_mode_parse(const char *s);

/* Resolves whether ANSI color should actually be emitted:
 *   TN_COLOR_ALWAYS -> 1
 *   TN_COLOR_NEVER  -> 0
 *   TN_COLOR_AUTO   -> is_tty && NO_COLOR is unset/empty (no-color.org)
 * is_tty/no_color_env are passed in rather than probed internally so this
 * stays pure and unit-testable without a real terminal. */
int tn_color_resolve(TnColorMode mode, int is_tty, const char *no_color_env);

/* ANSI SGR codes used by the CLI. */
#define TN_ANSI_RESET   "\x1b[0m"
#define TN_ANSI_BOLD    "\x1b[1m"
#define TN_ANSI_DIM     "\x1b[2m"
#define TN_ANSI_GREEN   "\x1b[32m"
#define TN_ANSI_CYAN    "\x1b[36m"
#define TN_ANSI_YELLOW  "\x1b[33m"

/* Returns ansi_code if enabled, else "" — safe to splice into printf/fprintf
 * unconditionally instead of #ifdef-ing every color call site. */
const char *tn_color(int enabled, const char *ansi_code);

#endif /* TN_CLI_COLOR_H */
