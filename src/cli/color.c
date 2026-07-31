/*
 * Phase 22.3 — src/cli/color.c
 * See include/cli/color.h for responsibility.
 */

#include "cli/color.h"
#include <string.h>
#include <strings.h>

TnColorMode tn_color_mode_parse(const char *s) {
    if (!s) return TN_COLOR_AUTO;
    if (strcasecmp(s, "always") == 0) return TN_COLOR_ALWAYS;
    if (strcasecmp(s, "never") == 0)  return TN_COLOR_NEVER;
    return TN_COLOR_AUTO; /* "auto" or anything unrecognized */
}

int tn_color_resolve(TnColorMode mode, int is_tty, const char *no_color_env) {
    if (mode == TN_COLOR_ALWAYS) return 1;
    if (mode == TN_COLOR_NEVER)  return 0;
    /* AUTO: respect NO_COLOR (any non-empty value disables) and require a TTY. */
    if (no_color_env && no_color_env[0] != '\0') return 0;
    return is_tty ? 1 : 0;
}

const char *tn_color(int enabled, const char *ansi_code) {
    return enabled ? ansi_code : "";
}
