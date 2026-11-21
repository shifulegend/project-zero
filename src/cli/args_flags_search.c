/*
 * src/cli/args_flags_search.c
 *
 * Command-line flags: best-of and tree-search (MCTS) flags and the reasoning judge. One group of
 * parse_args() (args.c); each branch is unchanged, and tests/test_args_golden.c compares every
 * flag's parse result with a golden table.
 */

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "api/chat_compile.h"
#include "api/server_config.h"
#include "cli/arg_values.h"
#include "cli/args.h"
#include "cli/args_table.h"
#include "core/log.h"
#include "kv_cache/kv_quant_ops.h"
#include "math/rope.h"
#include "reasoning/best_of.h"
#include "transformer/attn_tiled.h"
#include "update/update_check.h"

static const ArgFlagSpec k_flags[] = {
    {"--best-of", offsetof(CliArgs, sampling.best_of), ARG_K_INT, ARG_R_CLOSED, 1, BEST_OF_N_MAX,
     "--best-of must be 1.." TN_ARG_STR(BEST_OF_N_MAX)},
    {"--best-of-max", offsetof(CliArgs, sampling.best_of_max), ARG_K_INT, ARG_R_CLOSED, 1,
     BEST_OF_N_MAX, "--best-of-max must be 1.." TN_ARG_STR(BEST_OF_N_MAX)},
    {"--mcts", offsetof(CliArgs, search.mcts.simulations), ARG_K_INT, ARG_R_CLOSED, 0, 4096,
     "--mcts must be 0..4096"},
    {"--mcts-max", offsetof(CliArgs, search.mcts_max), ARG_K_INT, ARG_R_CLOSED, 0, 4096,
     "--mcts-max must be 0..4096"},
    {"--mcts-expand", offsetof(CliArgs, search.mcts.expand), ARG_K_INT, ARG_R_CLOSED, 1, 16,
     "--mcts-expand must be 1..16"},
    {"--mcts-depth", offsetof(CliArgs, search.mcts.max_depth), ARG_K_INT, ARG_R_CLOSED, 1, 256,
     "--mcts-depth must be 1..256"},
    {"--mcts-step-tokens", offsetof(CliArgs, search.mcts.step_tokens), ARG_K_INT, ARG_R_CLOSED, 1,
     4096, "--mcts-step-tokens must be 1..4096"},
    {"--mcts-c", offsetof(CliArgs, search.mcts.c_ucb), ARG_K_FLOAT, ARG_R_OPEN_LOW, 0, 10,
     "--mcts-c must be in (0, 10]"},
    {"--mcts-temperature", offsetof(CliArgs, search.mcts.temperature), ARG_K_FLOAT, ARG_R_OPEN_LOW,
     0, 2, "--mcts-temperature must be in (0, 2]"},
    {"--prm", offsetof(CliArgs, search.prm_path), ARG_K_STR, ARG_R_NONE, 0, 0, NULL},
    {"--prm-format", offsetof(CliArgs, search.prm_format), ARG_K_STR, ARG_R_NONE, 0, 0, NULL},
    {"--prm-good", offsetof(CliArgs, search.prm_good), ARG_K_STR, ARG_R_NONE, 0, 0, NULL},
    {"--prm-bad", offsetof(CliArgs, search.prm_bad), ARG_K_STR, ARG_R_NONE, 0, 0, NULL},
};

/* --mcts-delim <text>: "\n" in the argument means a newline. */
static TernaryError parse_mcts_delim(CliArgs *args, int argc, char **argv, int *ip, int *hit) {
    if (strcmp(argv[*ip], "--mcts-delim") != 0 || *ip + 1 >= argc) return TN_OK;
    const char *d = argv[++*ip];
    size_t k      = 0;
    for (; *d && k + 1 < sizeof(args->search.mcts_delim_buf); d++) {
        if (d[0] == '\\' && d[1] == 'n') {
            args->search.mcts_delim_buf[k++] = '\n';
            d++;
        } else
            args->search.mcts_delim_buf[k++] = *d;
    }
    args->search.mcts_delim_buf[k] = '\0';
    if (k == 0) {
        TN_LOG_ERR("args_flags_search", "Error: --mcts-delim must not be empty\n");
        return TN_ERR_INVALID_CONFIG;
    }
    args->search.mcts.step_delim = args->search.mcts_delim_buf;
    *hit                         = 1;
    return TN_OK;
}

TernaryError args_parse_search(CliArgs *args, int argc, char **argv, int *ip, int *handled) {
    int i = *ip, hit = 0;
    TernaryError err = args_table_parse(k_flags, sizeof(k_flags) / sizeof(k_flags[0]),
                                        "args_flags_search", args, argc, argv, &i, &hit);
    if (err == TN_OK && !hit) err = parse_mcts_delim(args, argc, argv, &i, &hit);
    return args_group_result(err, hit, i, ip, handled);
}
