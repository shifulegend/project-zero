#ifndef PZ_CLI_PROGRAM_NAME_H
#define PZ_CLI_PROGRAM_NAME_H

/*
 * cli/program_name — the executable's name and the Phase 39 transition shim.
 *
 * The binary is `pz` (Phase 39, 2026-10-21). For one release `make` and the
 * release tarball also ship `adaptive_ai_engine` as a symlink to it; running
 * through that name prints a one-line deprecation notice to stderr.
 */

#define PZ_PROGRAM_NAME        "pz"
#define PZ_LEGACY_PROGRAM_NAME "adaptive_ai_engine"

/* Final path component of argv0 ("" for NULL). */
const char *pz_program_basename(const char *argv0);

/* 1 when argv0 names the legacy executable, 0 otherwise. */
int pz_is_legacy_invocation(const char *argv0);

/* Prints the deprecation notice to stderr when invoked through the legacy name. */
void pz_warn_if_legacy_invocation(const char *argv0);

#endif /* PZ_CLI_PROGRAM_NAME_H */
