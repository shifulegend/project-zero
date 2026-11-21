/*
 * src/cli/args_flags_memory.c
 *
 * Command-line flags: perplexity, RoPE, attention kernel, KV cache and prompt-cache flags. One
 * group of parse_args() (args.c); each branch is unchanged, and tests/test_args_golden.c compares
 * every flag's parse result with a golden table.
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
#include "reasoning/best_of.h"
#include "transformer/attn_tiled.h"
#include "update/update_check.h"

TernaryError args_parse_memory(CliArgs *args, int argc, char **argv, int *ip, int *handled) {
    int i = *ip;
    if (strcmp(argv[i], "--perplexity") == 0 && i + 1 < argc) {
        args->eval.perplexity_file = argv[++i];
    } else if (strcmp(argv[i], "--ppl-ctx") == 0 && i + 1 < argc) {
        ARG_INT(args->eval.ppl_ctx);
        if (args->eval.ppl_ctx < 4) {
            TN_LOG_ERR("args_flags_memory", "Error: --ppl-ctx must be >= 4\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--ppl-chunks") == 0 && i + 1 < argc) {
        ARG_INT(args->eval.ppl_chunks);
        if (args->eval.ppl_chunks < 0) {
            TN_LOG_ERR("args_flags_memory", "Error: --ppl-chunks must be >= 0\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--rope-layout") == 0 && i + 1 < argc) {
        const char *m          = argv[++i];
        args->perf.rope_layout = strcmp(m, "auto") == 0 ? -1 : rope_layout_parse(m);
        if (strcmp(m, "auto") != 0 && args->perf.rope_layout < 0) {
            TN_LOG_ERR("args_flags_memory", "Error: --rope-layout must be auto, norm or neox\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--attn-kernel") == 0 && i + 1 < argc) {
        args->perf.attn_kernel = argv[++i];
        if (attn_kernel_parse(args->perf.attn_kernel) < 0) {
            TN_LOG_ERR("args_flags_memory", "Error: --attn-kernel must be standard or tiled\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--attn-tile") == 0 && i + 1 < argc) {
        ARG_INT(args->perf.attn_tile);
        if (args->perf.attn_tile < 0) {
            TN_LOG_ERR("args_flags_memory", "Error: --attn-tile must be >= 0\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--attn-parallel-min-ctx") == 0 && i + 1 < argc) {
        ARG_INT(args->perf.attn_parallel_min_ctx);
        if (args->perf.attn_parallel_min_ctx < 0) {
            TN_LOG_ERR("args_flags_memory", "Error: --attn-parallel-min-ctx must be >= 0\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--kv-type") == 0 && i + 1 < argc) {
        args->perf.kv_type = kv_type_parse(argv[++i]);
        if (args->perf.kv_type < 0) {
            TN_LOG_ERR("args_flags_memory", "Error: --kv-type must be f32, i8 or i4\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--kv-offload") == 0 && i + 1 < argc) {
        args->perf.kv_offload_dir = argv[++i];
    } else if (strcmp(argv[i], "--ctx-size") == 0 && i + 1 < argc) {
        ARG_INT(args->perf.ctx_size);
        if (args->perf.ctx_size < 0) {
            TN_LOG_ERR("args_flags_memory", "Error: --ctx-size must be >= 0\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--kv-ram-fraction") == 0 && i + 1 < argc) {
        ARG_NUM(args->perf.kv_ram_fraction);
        if (!(args->perf.kv_ram_fraction > 0.0f && args->perf.kv_ram_fraction <= 0.95f)) {
            TN_LOG_ERR("args_flags_memory", "Error: --kv-ram-fraction must be in (0, 0.95]\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--prompt-cache-mb") == 0 && i + 1 < argc) {
        ARG_INT(args->perf.prompt_cache_mb);
        if (args->perf.prompt_cache_mb < 0) {
            TN_LOG_ERR("args_flags_memory", "Error: --prompt-cache-mb must be >= 0\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--prompt-cache-min-tokens") == 0 && i + 1 < argc) {
        ARG_INT(args->perf.prompt_cache_min_tokens);
        if (args->perf.prompt_cache_min_tokens < 1) {
            TN_LOG_ERR("args_flags_memory", "Error: --prompt-cache-min-tokens must be >= 1\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--prompt-cache") == 0 && i + 1 < argc) {
        const char *mode = argv[++i];
        if (strcmp(mode, "on") == 0)
            args->perf.prompt_cache = true;
        else if (strcmp(mode, "off") == 0)
            args->perf.prompt_cache = false;
        else {
            TN_LOG_ERR("args_flags_memory", "Error: --prompt-cache must be on or off\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else {
        return TN_OK; /* not one of this group's flags */
    }
    *ip      = i;
    *handled = 1;
    return TN_OK;
}
