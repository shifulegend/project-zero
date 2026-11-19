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
#include "kv_cache/kv_quant_ops.h"
#include "math/rope.h"
#include "math/simd_dispatch.h"
#include "reasoning/best_of.h"
#include "transformer/attn_tiled.h"
#include "update/update_check.h"

TernaryError args_parse_search(CliArgs *args, int argc, char **argv, int *ip, int *handled) {
    int i = *ip;
    if (strcmp(argv[i], "--best-of") == 0 && i + 1 < argc) {
        ARG_INT(args->best_of);
        if (args->best_of < 1 || args->best_of > BEST_OF_N_MAX) {
            fprintf(stderr, "Error: --best-of must be 1..%d\n", BEST_OF_N_MAX);
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--mcts") == 0 && i + 1 < argc) {
        ARG_INT(args->mcts.simulations);
        if (args->mcts.simulations < 0 || args->mcts.simulations > 4096) {
            fprintf(stderr, "Error: --mcts must be 0..4096\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--mcts-max") == 0 && i + 1 < argc) {
        ARG_INT(args->mcts_max);
        if (args->mcts_max < 0 || args->mcts_max > 4096) {
            fprintf(stderr, "Error: --mcts-max must be 0..4096\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--mcts-expand") == 0 && i + 1 < argc) {
        ARG_INT(args->mcts.expand);
        if (args->mcts.expand < 1 || args->mcts.expand > 16) {
            fprintf(stderr, "Error: --mcts-expand must be 1..16\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--mcts-depth") == 0 && i + 1 < argc) {
        ARG_INT(args->mcts.max_depth);
        if (args->mcts.max_depth < 1 || args->mcts.max_depth > 256) {
            fprintf(stderr, "Error: --mcts-depth must be 1..256\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--mcts-step-tokens") == 0 && i + 1 < argc) {
        ARG_INT(args->mcts.step_tokens);
        if (args->mcts.step_tokens < 1 || args->mcts.step_tokens > 4096) {
            fprintf(stderr, "Error: --mcts-step-tokens must be 1..4096\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--mcts-c") == 0 && i + 1 < argc) {
        ARG_NUM(args->mcts.c_ucb);
        if (!(args->mcts.c_ucb > 0.0f && args->mcts.c_ucb <= 10.0f)) {
            fprintf(stderr, "Error: --mcts-c must be in (0, 10]\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--mcts-temperature") == 0 && i + 1 < argc) {
        ARG_NUM(args->mcts.temperature);
        if (!(args->mcts.temperature > 0.0f && args->mcts.temperature <= 2.0f)) {
            fprintf(stderr, "Error: --mcts-temperature must be in (0, 2]\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--mcts-delim") == 0 && i + 1 < argc) {
        /* "\n" in the argument means a newline */
        const char *d = argv[++i];
        size_t k      = 0;
        for (; *d && k + 1 < sizeof(args->mcts_delim_buf); d++) {
            if (d[0] == '\\' && d[1] == 'n') {
                args->mcts_delim_buf[k++] = '\n';
                d++;
            } else
                args->mcts_delim_buf[k++] = *d;
        }
        args->mcts_delim_buf[k] = '\0';
        if (k == 0) {
            fprintf(stderr, "Error: --mcts-delim must not be empty\n");
            return TN_ERR_INVALID_CONFIG;
        }
        args->mcts.step_delim = args->mcts_delim_buf;
    } else if (strcmp(argv[i], "--prm") == 0 && i + 1 < argc) {
        args->prm_path = argv[++i];
    } else if (strcmp(argv[i], "--prm-format") == 0 && i + 1 < argc) {
        args->prm_format = argv[++i];
    } else if (strcmp(argv[i], "--prm-good") == 0 && i + 1 < argc) {
        args->prm_good = argv[++i];
    } else if (strcmp(argv[i], "--prm-bad") == 0 && i + 1 < argc) {
        args->prm_bad = argv[++i];
    } else if (strcmp(argv[i], "--best-of-max") == 0 && i + 1 < argc) {
        ARG_INT(args->best_of_max);
        if (args->best_of_max < 1 || args->best_of_max > BEST_OF_N_MAX) {
            fprintf(stderr, "Error: --best-of-max must be 1..%d\n", BEST_OF_N_MAX);
            return TN_ERR_INVALID_CONFIG;
        }
    } else {
        return TN_OK; /* not one of this group's flags */
    }
    *ip      = i;
    *handled = 1;
    return TN_OK;
}
