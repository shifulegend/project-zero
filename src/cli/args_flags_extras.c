/*
 * src/cli/args_flags_extras.c
 *
 * Command-line flags: draft model, LoRA, colour and update-check flags. One group of parse_args() (args.c); each branch is unchanged, and
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

TernaryError args_parse_extras(CliArgs *args, int argc, char **argv, int *ip, int *handled) {
    int i = *ip;
    if (strcmp(argv[i], "--draft-model") == 0 && i + 1 < argc) {
        args->draft_model_path = argv[++i];
    } else if (strcmp(argv[i], "--spec-length") == 0 && i + 1 < argc) {
        ARG_INT(args->spec_length);
        if (args->spec_length <= 0) {
            TN_LOG_ERR("args_flags_extras", "Error: --spec-length must be a positive integer\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--lora") == 0 && i + 1 < argc) {
        args->lora_path = argv[++i];
    } else if (strcmp(argv[i], "--color") == 0 && i + 1 < argc) {
        const char *mode = argv[++i];
        if (strcmp(mode, "auto") != 0 && strcmp(mode, "always") != 0 &&
            strcmp(mode, "never") != 0) {
            TN_LOG_ERR("args_flags_extras", "Error: --color must be one of: auto, always, never\n");
            return TN_ERR_INVALID_CONFIG;
        }
        args->color_mode = tn_color_mode_parse(mode);
    } else if (strcmp(argv[i], "--check-update") == 0) {
        args->check_update = true;
    } else if (strcmp(argv[i], "--no-update-check") == 0) {
        args->no_update_check = true;
    } else if (strcmp(argv[i], "--update-repo") == 0 && i + 1 < argc) {
        args->update_repo = argv[++i];
        if (!update_repo_valid(args->update_repo)) {
            TN_LOG_ERR("args_flags_extras", "Error: --update-repo must be owner/name\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--update-api") == 0 && i + 1 < argc) {
        args->update_api = argv[++i];
        if (!update_api_base_valid(args->update_api)) {
            TN_LOG_ERR("args_flags_extras", "Error: --update-api must be an https:// URL\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--update-interval") == 0 && i + 1 < argc) {
        ARG_INT(args->update_interval_h);
        if (args->update_interval_h < 0 || args->update_interval_h > 24 * 365) {
            TN_LOG_ERR("args_flags_extras", "Error: --update-interval must be 0-8760 hours\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--update-channel") == 0 && i + 1 < argc) {
        args->update_channel = update_channel_parse(argv[++i]);
        if (args->update_channel < 0) {
            TN_LOG_ERR("args_flags_extras",
                       "Error: --update-channel must be auto, stable or prerelease\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else {
        return TN_OK; /* not one of this group's flags */
    }
    *ip      = i;
    *handled = 1;
    return TN_OK;
}
