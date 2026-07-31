#ifndef TN_CLI_PROGRESS_H
#define TN_CLI_PROGRESS_H

/*
 * Phase 22.3 — Coarse-grained model-load progress indicator.
 * No fine-grained byte callback is threaded into the loader internals (that
 * would be a loader-logic change, out of scope for UI polish) — this marks
 * named milestones (mmap opened, weights loaded, tokenizer loaded, ...).
 */

#include <stddef.h>

/* Pure formatter — no TTY/stream dependency, fully unit-testable.
 * is_tty=0:  "[2/4] Loading weights...\n"
 * is_tty=1:  "\r[2/4] Loading weights...\x1b[K" (in-place update, no newline;
 *            \x1b[K clears any leftover text from a longer previous line).
 * Returns the number of bytes written (excluding the NUL terminator),
 * truncated safely to cap. */
size_t tn_progress_format(char *buf, size_t cap, int stage, int total_stages,
                           const char *label, int is_tty);

/* Writes the formatted line for this stage to stderr. */
void tn_progress_stage(int stage, int total_stages, const char *label, int is_tty);

/* Terminates the progress display (moves past the in-place line). No-op
 * when is_tty is false, since non-TTY output already ends each stage with
 * its own newline. */
void tn_progress_done(int is_tty);

#endif /* TN_CLI_PROGRESS_H */
