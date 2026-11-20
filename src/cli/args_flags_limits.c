/*
 * src/cli/args_flags_limits.c
 *
 * Command-line flags: request queue, rate, timeout, connection and prompt-size limits of the API
 * server. One group of parse_args() (args.c), reached from args_parse_serving(); the flags are a
 * table of (name, field, range, message), and tests/test_args_golden.c compares every flag's parse
 * result and message with a golden table.
 */

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "cli/arg_values.h"
#include "cli/args.h"
#include "cli/args_groups.h"
#include "core/log.h"

typedef struct {
    const char *flag;
    size_t offset; /* of the int field inside CliServer */
    int lo, hi;
    const char *rule; /* "Error: <flag> must be <rule>" */
} LimitFlag;

static const LimitFlag k_limit_flags[] = {
    {"--queue-size", offsetof(CliServer, queue_size), 0, 2147483647, ">= 0"},
    {"--queue-timeout", offsetof(CliServer, queue_timeout_s), 1, 2147483647, ">= 1 second"},
    {"--rate-limit", offsetof(CliServer, rate_limit), 0, 100000, "0..100000 requests per second"},
    {"--client-timeout", offsetof(CliServer, client_timeout_s), 0, 86400, "0..86400 seconds"},
    {"--max-prompt-bytes", offsetof(CliServer, max_prompt_bytes), 0, 1073741824,
     "0..1073741824 (0 = no cap)"},
    {"--max-connections", offsetof(CliServer, max_connections), 0, 1048576, "0..1048576"},
};

TernaryError args_parse_limits(CliArgs *args, int argc, char **argv, int *ip, int *handled) {
    int i = *ip;
    for (size_t k = 0; k < sizeof(k_limit_flags) / sizeof(k_limit_flags[0]); k++) {
        const LimitFlag *f = &k_limit_flags[k];
        if (strcmp(argv[i], f->flag) != 0 || i + 1 >= argc) continue;
        int *dst = (int *)((char *)&args->server + f->offset);
        ARG_INT(*dst);
        if (*dst < f->lo || *dst > f->hi) {
            TN_LOG_ERR("args_flags_limits", "Error: %s must be %s\n", f->flag, f->rule);
            return TN_ERR_INVALID_CONFIG;
        }
        *ip      = i;
        *handled = 1;
        return TN_OK;
    }
    return TN_OK; /* not one of this group's flags */
}
