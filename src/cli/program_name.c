/*
 * cli/program_name.c — see cli/program_name.h.
 */
#include "cli/program_name.h"

#include <stdio.h>
#include <string.h>

const char *pz_program_basename(const char *argv0) {
    if (!argv0) return "";
    const char *base = argv0;
    for (const char *p = argv0; *p; p++)
        if (*p == '/' || *p == '\\') base = p + 1;
    return base;
}

int pz_is_legacy_invocation(const char *argv0) {
    const char *b = pz_program_basename(argv0);
    size_t n = strlen(PZ_LEGACY_PROGRAM_NAME);
    /* Windows builds see "adaptive_ai_engine.exe". */
    return strncmp(b, PZ_LEGACY_PROGRAM_NAME, n) == 0 && (b[n] == '\0' || strcmp(b + n, ".exe") == 0);
}

void pz_warn_if_legacy_invocation(const char *argv0) {
    if (pz_is_legacy_invocation(argv0))
        fprintf(stderr, "note: '%s' has been renamed to '%s'; the old name will be removed "
                        "in the next release.\n", PZ_LEGACY_PROGRAM_NAME, PZ_PROGRAM_NAME);
}
