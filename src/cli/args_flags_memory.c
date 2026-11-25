/*
 * src/cli/args_flags_memory.c
 *
 * Command-line flags: perplexity, RoPE, attention kernel, KV cache and prompt-cache flags. One
 * group of parse_args() (args.c); each branch is unchanged, and tests/test_args_golden.c compares
 * every flag's parse result with a golden table.
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
    {"--perplexity", offsetof(CliArgs, eval.perplexity_file), ARG_K_STR, ARG_R_NONE, 0, 0, NULL},
    {"--kv-offload", offsetof(CliArgs, perf.kv_offload_dir), ARG_K_STR, ARG_R_NONE, 0, 0, NULL},
    {"--ppl-ctx", offsetof(CliArgs, eval.ppl_ctx), ARG_K_INT, ARG_R_CLOSED, 4, INT_MAX,
     "--ppl-ctx must be >= 4"},
    {"--ppl-chunks", offsetof(CliArgs, eval.ppl_chunks), ARG_K_INT, ARG_R_CLOSED, 0, INT_MAX,
     "--ppl-chunks must be >= 0"},
    {"--attn-tile", offsetof(CliArgs, perf.attn_tile), ARG_K_INT, ARG_R_CLOSED, 0, INT_MAX,
     "--attn-tile must be >= 0"},
    {"--attn-parallel-min-ctx", offsetof(CliArgs, perf.attn_parallel_min_ctx), ARG_K_INT,
     ARG_R_CLOSED, 0, INT_MAX, "--attn-parallel-min-ctx must be >= 0"},
    {"--ctx-size", offsetof(CliArgs, perf.ctx_size), ARG_K_INT, ARG_R_CLOSED, 0, INT_MAX,
     "--ctx-size must be >= 0"},
    {"--prompt-cache-mb", offsetof(CliArgs, perf.prompt_cache_mb), ARG_K_INT, ARG_R_CLOSED, 0,
     INT_MAX, "--prompt-cache-mb must be >= 0"},
    {"--prompt-cache-min-tokens", offsetof(CliArgs, perf.prompt_cache_min_tokens), ARG_K_INT,
     ARG_R_CLOSED, 1, INT_MAX, "--prompt-cache-min-tokens must be >= 1"},
    {"--kv-ram-fraction", offsetof(CliArgs, perf.kv_ram_fraction), ARG_K_FLOAT, ARG_R_OPEN_LOW, 0,
     0.95, "--kv-ram-fraction must be in (0, 0.95]"},
};

static TernaryError fail(const char *msg) {
    TN_LOG_ERR("args_flags_memory", "Error: %s\n", msg);
    return TN_ERR_INVALID_CONFIG;
}

static TernaryError parse_rope_layout(CliArgs *args, const char *m) {
    args->perf.rope_layout = strcmp(m, "auto") == 0 ? -1 : rope_layout_parse(m);
    if (strcmp(m, "auto") != 0 && args->perf.rope_layout < 0)
        return fail("--rope-layout must be auto, norm or neox");
    return TN_OK;
}

static TernaryError parse_attn_kernel(CliArgs *args, const char *m) {
    args->perf.attn_kernel = (char *)m;
    if (attn_kernel_parse(m) < 0) return fail("--attn-kernel must be standard or tiled");
    return TN_OK;
}

static TernaryError parse_kv_type(CliArgs *args, const char *m) {
    args->perf.kv_type = kv_type_parse(m);
    if (args->perf.kv_type < 0) return fail("--kv-type must be f32, i8 or i4");
    return TN_OK;
}

static TernaryError parse_prompt_cache(CliArgs *args, const char *mode) {
    if (strcmp(mode, "on") == 0)
        args->perf.prompt_cache = true;
    else if (strcmp(mode, "off") == 0)
        args->perf.prompt_cache = false;
    else
        return fail("--prompt-cache must be on or off");
    return TN_OK;
}

static const struct {
    const char *flag;
    TernaryError (*parse)(CliArgs *args, const char *value);
} k_handlers[] = {
    {"--rope-layout", parse_rope_layout},
    {"--attn-kernel", parse_attn_kernel},
    {"--kv-type", parse_kv_type},
    {"--prompt-cache", parse_prompt_cache},
};

static TernaryError parse_handler(CliArgs *args, int argc, char **argv, int *ip, int *hit) {
    for (size_t k = 0; k < sizeof(k_handlers) / sizeof(k_handlers[0]); k++) {
        if (strcmp(argv[*ip], k_handlers[k].flag) != 0 || *ip + 1 >= argc) continue;
        *hit = 1;
        return k_handlers[k].parse(args, argv[++*ip]);
    }
    return TN_OK;
}

TernaryError args_parse_memory(CliArgs *args, int argc, char **argv, int *ip, int *handled) {
    int i = *ip, hit = 0;
    TernaryError err = args_table_parse(k_flags, sizeof(k_flags) / sizeof(k_flags[0]),
                                        "args_flags_memory", args, argc, argv, &i, &hit);
    if (err == TN_OK && !hit) err = parse_handler(args, argc, argv, &i, &hit);
    return args_group_result(err, hit, i, ip, handled);
}
