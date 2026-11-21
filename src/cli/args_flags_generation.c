/*
 * src/cli/args_flags_generation.c
 *
 * Command-line flags: model and prompt inputs, sampling, penalties, stop strings and the watermark.
 * One group of parse_args() (args.c); each branch is unchanged, and tests/test_args_golden.c
 * compares every flag's parse result with a golden table.
 */

#include <limits.h>
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
    {"--model", offsetof(CliArgs, input.model_path), ARG_K_STR, ARG_R_NONE, 0, 0, NULL},
    {"--tokenizer", offsetof(CliArgs, input.tokenizer_path), ARG_K_STR, ARG_R_NONE, 0, 0, NULL},
    {"--prompt", offsetof(CliArgs, input.prompt), ARG_K_STR, ARG_R_NONE, 0, 0, NULL},
    {"--image", offsetof(CliArgs, input.image_path), ARG_K_STR, ARG_R_NONE, 0, 0, NULL},
    {"--vision", offsetof(CliArgs, input.vision_path), ARG_K_STR, ARG_R_NONE, 0, 0, NULL},
    {"--proj", offsetof(CliArgs, input.proj_path), ARG_K_STR, ARG_R_NONE, 0, 0, NULL},
    {"--temperature", offsetof(CliArgs, sampling.temperature), ARG_K_FLOAT, ARG_R_NONE, 0, 0, NULL},
    {"--top-p", offsetof(CliArgs, sampling.top_p), ARG_K_FLOAT, ARG_R_NONE, 0, 0, NULL},
    {"--max-tokens", offsetof(CliArgs, sampling.max_tokens), ARG_K_INT, ARG_R_NONE, 0, 0, NULL},
    {"--threads", offsetof(CliArgs, perf.num_threads), ARG_K_INT, ARG_R_NONE, 0, 0, NULL},
    {"--seed", offsetof(CliArgs, sampling.seed), ARG_K_INT, ARG_R_NONE, 0, 0, NULL},
    {"--top-k", offsetof(CliArgs, sampling.top_k), ARG_K_INT, ARG_R_NONE, 0, 0, NULL},
    {"--min-p", offsetof(CliArgs, sampling.min_p), ARG_K_FLOAT, ARG_R_NONE, 0, 0, NULL},
    {"--repeat-penalty", offsetof(CliArgs, sampling.repeat_penalty), ARG_K_FLOAT, ARG_R_NONE, 0, 0,
     NULL},
    {"--presence-penalty", offsetof(CliArgs, sampling.presence_penalty), ARG_K_FLOAT, ARG_R_NONE, 0,
     0, NULL},
    {"--frequency-penalty", offsetof(CliArgs, sampling.frequency_penalty), ARG_K_FLOAT, ARG_R_NONE,
     0, 0, NULL},
    {"--repeat-last-n", offsetof(CliArgs, sampling.repeat_last_n), ARG_K_INT, ARG_R_NONE, 0, 0,
     NULL},
    {"--system-prompt", offsetof(CliArgs, input.system_prompt), ARG_K_STR, ARG_R_NONE, 0, 0, NULL},
    {"--chat-template-file", offsetof(CliArgs, input.chat_template_file), ARG_K_STR, ARG_R_NONE, 0,
     0, NULL},
    {"--watermark-key", offsetof(CliArgs, watermark.watermark_key), ARG_K_STR, ARG_R_NONE, 0, 0,
     NULL},
    {"--watermark-gamma", offsetof(CliArgs, watermark.watermark_gamma), ARG_K_FLOAT, ARG_R_NONE, 0,
     0, NULL},
    {"--watermark-delta", offsetof(CliArgs, watermark.watermark_delta), ARG_K_FLOAT, ARG_R_NONE, 0,
     0, NULL},
    {"--no-request-template", offsetof(CliArgs, server.no_request_template), ARG_K_SET, ARG_R_NONE,
     0, 0, NULL},
    {"--no-config-write", offsetof(CliArgs, server.no_config_write), ARG_K_SET, ARG_R_NONE, 0, 0,
     NULL},
    {"--watermark", offsetof(CliArgs, watermark.enabled), ARG_K_SET, ARG_R_NONE, 0, 0, NULL},
    {"--max-tokens-limit", offsetof(CliArgs, sampling.max_tokens_limit), ARG_K_INT, ARG_R_CLOSED, 0,
     INT_MAX, "--max-tokens-limit must be >= 0"},
};

/* --stop <text>, repeatable up to GEN_MAX_STOP. */
static TernaryError parse_stop(CliArgs *args, int argc, char **argv, int *ip, int *hit) {
    if (strcmp(argv[*ip], "--stop") != 0 || *ip + 1 >= argc) return TN_OK;
    if (args->sampling.n_stop >= GEN_MAX_STOP) {
        TN_LOG_ERR("args_flags_generation", "Error: too many --stop values (max %d)\n",
                   GEN_MAX_STOP);
        return TN_ERR_INVALID_CONFIG;
    }
    args->sampling.stop[args->sampling.n_stop++] = argv[++*ip];
    *hit                                         = 1;
    return TN_OK;
}

TernaryError args_parse_generation(CliArgs *args, int argc, char **argv, int *ip, int *handled) {
    int i = *ip, hit = 0;
    TernaryError err = args_table_parse(k_flags, sizeof(k_flags) / sizeof(k_flags[0]),
                                        "args_flags_generation", args, argc, argv, &i, &hit);
    if (err == TN_OK && !hit) err = parse_stop(args, argc, argv, &i, &hit);
    return args_group_result(err, hit, i, ip, handled);
}
