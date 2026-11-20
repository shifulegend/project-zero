/*
 * src/cli/args_flags_search.c
 *
 * Command-line flags: best-of and tree-search (MCTS) flags and the reasoning judge. One group of parse_args() (args.c); each branch is unchanged, and
 * tests/test_args_golden.c compares every flag's parse result with a golden table.
 */

#include <stdio.h>
#include <string.h>

#include "api/chat_compile.h"
#include "api/server_config.h"
#include "cli/arg_values.h"
#include "cli/args.h"
#include "core/log.h"
#include "kv_cache/kv_quant_ops.h"
#include "math/rope.h"
#include "math/simd_dispatch.h"
#include "reasoning/best_of.h"
#include "transformer/attn_tiled.h"
#include "update/update_check.h"

TernaryError args_parse_search(CliArgs *args, int argc, char **argv, int *ip, int *handled) {
    int i = *ip;
    if (strcmp(argv[i], "--best-of") == 0 && i + 1 < argc) {
        ARG_INT(args->sampling.best_of);
        if (args->sampling.best_of < 1 || args->sampling.best_of > BEST_OF_N_MAX) {
            TN_LOG_ERR("args_flags_search", "Error: --best-of must be 1..%d\n", BEST_OF_N_MAX);
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--mcts") == 0 && i + 1 < argc) {
        ARG_INT(args->search.mcts.simulations);
        if (args->search.mcts.simulations < 0 || args->search.mcts.simulations > 4096) {
            TN_LOG_ERR("args_flags_search", "Error: --mcts must be 0..4096\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--mcts-max") == 0 && i + 1 < argc) {
        ARG_INT(args->search.mcts_max);
        if (args->search.mcts_max < 0 || args->search.mcts_max > 4096) {
            TN_LOG_ERR("args_flags_search", "Error: --mcts-max must be 0..4096\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--mcts-expand") == 0 && i + 1 < argc) {
        ARG_INT(args->search.mcts.expand);
        if (args->search.mcts.expand < 1 || args->search.mcts.expand > 16) {
            TN_LOG_ERR("args_flags_search", "Error: --mcts-expand must be 1..16\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--mcts-depth") == 0 && i + 1 < argc) {
        ARG_INT(args->search.mcts.max_depth);
        if (args->search.mcts.max_depth < 1 || args->search.mcts.max_depth > 256) {
            TN_LOG_ERR("args_flags_search", "Error: --mcts-depth must be 1..256\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--mcts-step-tokens") == 0 && i + 1 < argc) {
        ARG_INT(args->search.mcts.step_tokens);
        if (args->search.mcts.step_tokens < 1 || args->search.mcts.step_tokens > 4096) {
            TN_LOG_ERR("args_flags_search", "Error: --mcts-step-tokens must be 1..4096\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--mcts-c") == 0 && i + 1 < argc) {
        ARG_NUM(args->search.mcts.c_ucb);
        if (!(args->search.mcts.c_ucb > 0.0f && args->search.mcts.c_ucb <= 10.0f)) {
            TN_LOG_ERR("args_flags_search", "Error: --mcts-c must be in (0, 10]\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--mcts-temperature") == 0 && i + 1 < argc) {
        ARG_NUM(args->search.mcts.temperature);
        if (!(args->search.mcts.temperature > 0.0f && args->search.mcts.temperature <= 2.0f)) {
            TN_LOG_ERR("args_flags_search", "Error: --mcts-temperature must be in (0, 2]\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--mcts-delim") == 0 && i + 1 < argc) {
        /* "\n" in the argument means a newline */
        const char *d = argv[++i];
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
    } else if (strcmp(argv[i], "--prm") == 0 && i + 1 < argc) {
        args->search.prm_path = argv[++i];
    } else if (strcmp(argv[i], "--prm-format") == 0 && i + 1 < argc) {
        args->search.prm_format = argv[++i];
    } else if (strcmp(argv[i], "--prm-good") == 0 && i + 1 < argc) {
        args->search.prm_good = argv[++i];
    } else if (strcmp(argv[i], "--prm-bad") == 0 && i + 1 < argc) {
        args->search.prm_bad = argv[++i];
    } else if (strcmp(argv[i], "--best-of-max") == 0 && i + 1 < argc) {
        ARG_INT(args->sampling.best_of_max);
        if (args->sampling.best_of_max < 1 || args->sampling.best_of_max > BEST_OF_N_MAX) {
            TN_LOG_ERR("args_flags_search", "Error: --best-of-max must be 1..%d\n", BEST_OF_N_MAX);
            return TN_ERR_INVALID_CONFIG;
        }
    } else {
        return TN_OK; /* not one of this group's flags */
    }
    *ip      = i;
    *handled = 1;
    return TN_OK;
}
