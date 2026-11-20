/*
 * src/cli/args_flags_generation.c
 *
 * Command-line flags: model and prompt inputs, sampling, penalties, stop strings and the watermark. One group of parse_args() (args.c); each branch is unchanged, and
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

TernaryError args_parse_generation(CliArgs *args, int argc, char **argv, int *ip, int *handled) {
    int i = *ip;
    if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
        args->model_path = argv[++i];
    } else if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
        args->tokenizer_path = argv[++i];
    } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
        args->prompt = argv[++i];
    } else if (strcmp(argv[i], "--image") == 0 && i + 1 < argc) {
        args->image_path = argv[++i];
    } else if (strcmp(argv[i], "--vision") == 0 && i + 1 < argc) {
        args->vision_path = argv[++i];
    } else if (strcmp(argv[i], "--proj") == 0 && i + 1 < argc) {
        args->proj_path = argv[++i];
    } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
        ARG_NUM(args->temperature);
    } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
        ARG_NUM(args->top_p);
    } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
        ARG_INT(args->max_tokens);
    } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
        ARG_INT(args->num_threads);
    } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
        ARG_INT(args->seed);
    } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
        ARG_INT(args->top_k);
    } else if (strcmp(argv[i], "--min-p") == 0 && i + 1 < argc) {
        ARG_NUM(args->min_p);
    } else if (strcmp(argv[i], "--repeat-penalty") == 0 && i + 1 < argc) {
        ARG_NUM(args->repeat_penalty);
    } else if (strcmp(argv[i], "--presence-penalty") == 0 && i + 1 < argc) {
        ARG_NUM(args->presence_penalty);
    } else if (strcmp(argv[i], "--frequency-penalty") == 0 && i + 1 < argc) {
        ARG_NUM(args->frequency_penalty);
    } else if (strcmp(argv[i], "--repeat-last-n") == 0 && i + 1 < argc) {
        ARG_INT(args->repeat_last_n);
    } else if (strcmp(argv[i], "--system-prompt") == 0 && i + 1 < argc) {
        args->system_prompt = argv[++i];
    } else if (strcmp(argv[i], "--no-request-template") == 0) {
        args->no_request_template = true;
    } else if (strcmp(argv[i], "--no-config-write") == 0) {
        args->no_config_write = true;
    } else if (strcmp(argv[i], "--chat-template-file") == 0 && i + 1 < argc) {
        args->chat_template_file = argv[++i];
    } else if (strcmp(argv[i], "--stop") == 0 && i + 1 < argc) {
        if (args->n_stop >= GEN_MAX_STOP) {
            TN_LOG_ERR("args_flags_generation", "Error: too many --stop values (max %d)\n",
                       GEN_MAX_STOP);
            return TN_ERR_INVALID_CONFIG;
        }
        args->stop[args->n_stop++] = argv[++i];
    } else if (strcmp(argv[i], "--max-tokens-limit") == 0 && i + 1 < argc) {
        ARG_INT(args->max_tokens_limit);
        if (args->max_tokens_limit < 0) {
            TN_LOG_ERR("args_flags_generation", "Error: --max-tokens-limit must be >= 0\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--watermark") == 0) {
        args->watermark = true;
    } else if (strcmp(argv[i], "--watermark-key") == 0 && i + 1 < argc) {
        args->watermark_key = argv[++i];
    } else if (strcmp(argv[i], "--watermark-gamma") == 0 && i + 1 < argc) {
        ARG_NUM(args->watermark_gamma);
    } else if (strcmp(argv[i], "--watermark-delta") == 0 && i + 1 < argc) {
        ARG_NUM(args->watermark_delta);
    } else {
        return TN_OK; /* not one of this group's flags */
    }
    *ip      = i;
    *handled = 1;
    return TN_OK;
}
