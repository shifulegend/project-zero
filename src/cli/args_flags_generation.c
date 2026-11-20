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
        args->input.model_path = argv[++i];
    } else if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
        args->input.tokenizer_path = argv[++i];
    } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
        args->input.prompt = argv[++i];
    } else if (strcmp(argv[i], "--image") == 0 && i + 1 < argc) {
        args->input.image_path = argv[++i];
    } else if (strcmp(argv[i], "--vision") == 0 && i + 1 < argc) {
        args->input.vision_path = argv[++i];
    } else if (strcmp(argv[i], "--proj") == 0 && i + 1 < argc) {
        args->input.proj_path = argv[++i];
    } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
        ARG_NUM(args->sampling.temperature);
    } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
        ARG_NUM(args->sampling.top_p);
    } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
        ARG_INT(args->sampling.max_tokens);
    } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
        ARG_INT(args->perf.num_threads);
    } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
        ARG_INT(args->sampling.seed);
    } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
        ARG_INT(args->sampling.top_k);
    } else if (strcmp(argv[i], "--min-p") == 0 && i + 1 < argc) {
        ARG_NUM(args->sampling.min_p);
    } else if (strcmp(argv[i], "--repeat-penalty") == 0 && i + 1 < argc) {
        ARG_NUM(args->sampling.repeat_penalty);
    } else if (strcmp(argv[i], "--presence-penalty") == 0 && i + 1 < argc) {
        ARG_NUM(args->sampling.presence_penalty);
    } else if (strcmp(argv[i], "--frequency-penalty") == 0 && i + 1 < argc) {
        ARG_NUM(args->sampling.frequency_penalty);
    } else if (strcmp(argv[i], "--repeat-last-n") == 0 && i + 1 < argc) {
        ARG_INT(args->sampling.repeat_last_n);
    } else if (strcmp(argv[i], "--system-prompt") == 0 && i + 1 < argc) {
        args->input.system_prompt = argv[++i];
    } else if (strcmp(argv[i], "--no-request-template") == 0) {
        args->server.no_request_template = true;
    } else if (strcmp(argv[i], "--no-config-write") == 0) {
        args->server.no_config_write = true;
    } else if (strcmp(argv[i], "--chat-template-file") == 0 && i + 1 < argc) {
        args->input.chat_template_file = argv[++i];
    } else if (strcmp(argv[i], "--stop") == 0 && i + 1 < argc) {
        if (args->sampling.n_stop >= GEN_MAX_STOP) {
            TN_LOG_ERR("args_flags_generation", "Error: too many --stop values (max %d)\n",
                       GEN_MAX_STOP);
            return TN_ERR_INVALID_CONFIG;
        }
        args->sampling.stop[args->sampling.n_stop++] = argv[++i];
    } else if (strcmp(argv[i], "--max-tokens-limit") == 0 && i + 1 < argc) {
        ARG_INT(args->sampling.max_tokens_limit);
        if (args->sampling.max_tokens_limit < 0) {
            TN_LOG_ERR("args_flags_generation", "Error: --max-tokens-limit must be >= 0\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--watermark") == 0) {
        args->watermark.enabled = true;
    } else if (strcmp(argv[i], "--watermark-key") == 0 && i + 1 < argc) {
        args->watermark.watermark_key = argv[++i];
    } else if (strcmp(argv[i], "--watermark-gamma") == 0 && i + 1 < argc) {
        ARG_NUM(args->watermark.watermark_gamma);
    } else if (strcmp(argv[i], "--watermark-delta") == 0 && i + 1 < argc) {
        ARG_NUM(args->watermark.watermark_delta);
    } else {
        return TN_OK; /* not one of this group's flags */
    }
    *ip      = i;
    *handled = 1;
    return TN_OK;
}
