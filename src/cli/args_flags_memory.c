/*
 * src/cli/args_flags_memory.c
 *
 * Command-line flags: perplexity, RoPE, attention kernel, KV cache and prompt-cache flags. One group of parse_args() (args.c); each branch is unchanged, and
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

TernaryError args_parse_memory(CliArgs *args, int argc, char **argv, int *ip, int *handled) {
    int i = *ip;
    if (strcmp(argv[i], "--perplexity") == 0 && i + 1 < argc) {
        args->perplexity_file = argv[++i];
    } else if (strcmp(argv[i], "--ppl-ctx") == 0 && i + 1 < argc) {
        ARG_INT(args->ppl_ctx);
        if (args->ppl_ctx < 4) {
            fprintf(stderr, "Error: --ppl-ctx must be >= 4\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--ppl-chunks") == 0 && i + 1 < argc) {
        ARG_INT(args->ppl_chunks);
        if (args->ppl_chunks < 0) {
            fprintf(stderr, "Error: --ppl-chunks must be >= 0\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--rope-layout") == 0 && i + 1 < argc) {
        const char *m     = argv[++i];
        args->rope_layout = strcmp(m, "auto") == 0 ? -1 : rope_layout_parse(m);
        if (strcmp(m, "auto") != 0 && args->rope_layout < 0) {
            fprintf(stderr, "Error: --rope-layout must be auto, norm or neox\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--attn-kernel") == 0 && i + 1 < argc) {
        args->attn_kernel = argv[++i];
        if (attn_kernel_parse(args->attn_kernel) < 0) {
            fprintf(stderr, "Error: --attn-kernel must be standard or tiled\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--attn-tile") == 0 && i + 1 < argc) {
        ARG_INT(args->attn_tile);
        if (args->attn_tile < 0) {
            fprintf(stderr, "Error: --attn-tile must be >= 0\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--attn-parallel-min-ctx") == 0 && i + 1 < argc) {
        ARG_INT(args->attn_parallel_min_ctx);
        if (args->attn_parallel_min_ctx < 0) {
            fprintf(stderr, "Error: --attn-parallel-min-ctx must be >= 0\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--kv-type") == 0 && i + 1 < argc) {
        args->kv_type = kv_type_parse(argv[++i]);
        if (args->kv_type < 0) {
            fprintf(stderr, "Error: --kv-type must be f32, i8 or i4\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--kv-offload") == 0 && i + 1 < argc) {
        args->kv_offload_dir = argv[++i];
    } else if (strcmp(argv[i], "--ctx-size") == 0 && i + 1 < argc) {
        ARG_INT(args->ctx_size);
        if (args->ctx_size < 0) {
            fprintf(stderr, "Error: --ctx-size must be >= 0\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--kv-ram-fraction") == 0 && i + 1 < argc) {
        ARG_NUM(args->kv_ram_fraction);
        if (!(args->kv_ram_fraction > 0.0f && args->kv_ram_fraction <= 0.95f)) {
            fprintf(stderr, "Error: --kv-ram-fraction must be in (0, 0.95]\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--prompt-cache-mb") == 0 && i + 1 < argc) {
        ARG_INT(args->prompt_cache_mb);
        if (args->prompt_cache_mb < 0) {
            fprintf(stderr, "Error: --prompt-cache-mb must be >= 0\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--prompt-cache-min-tokens") == 0 && i + 1 < argc) {
        ARG_INT(args->prompt_cache_min_tokens);
        if (args->prompt_cache_min_tokens < 1) {
            fprintf(stderr, "Error: --prompt-cache-min-tokens must be >= 1\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--prompt-cache") == 0 && i + 1 < argc) {
        const char *mode = argv[++i];
        if (strcmp(mode, "on") == 0)
            args->prompt_cache = true;
        else if (strcmp(mode, "off") == 0)
            args->prompt_cache = false;
        else {
            fprintf(stderr, "Error: --prompt-cache must be on or off\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else {
        return TN_OK; /* not one of this group's flags */
    }
    *ip      = i;
    *handled = 1;
    return TN_OK;
}
