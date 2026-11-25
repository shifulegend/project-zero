/*
 * src/cli/args_flags_extras.c
 *
 * Command-line flags: draft model, LoRA, colour and update-check flags. One group of parse_args()
 * (args.c); each branch is unchanged, and tests/test_args_golden.c compares every flag's parse
 * result with a golden table.
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
    {"--draft-model", offsetof(CliArgs, input.draft_model_path), ARG_K_STR, ARG_R_NONE, 0, 0, NULL},
    {"--spec-length", offsetof(CliArgs, search.spec_length), ARG_K_INT, ARG_R_OPEN_LOW, 0,
     2147483647, "--spec-length must be a positive integer"},
    {"--lora", offsetof(CliArgs, input.lora_path), ARG_K_STR, ARG_R_NONE, 0, 0, NULL},
    {"--check-update", offsetof(CliArgs, update.check_update), ARG_K_SET, ARG_R_NONE, 0, 0, NULL},
    {"--no-update-check", offsetof(CliArgs, update.no_update_check), ARG_K_SET, ARG_R_NONE, 0, 0,
     NULL},
    {"--update-interval", offsetof(CliArgs, update.update_interval_h), ARG_K_INT, ARG_R_CLOSED, 0,
     24 * 365, "--update-interval must be 0-8760 hours"},
};

static TernaryError fail(const char *message) {
    TN_LOG_ERR("args_flags_extras", "Error: %s\n", message);
    return TN_ERR_INVALID_CONFIG;
}

static TernaryError set_color(CliArgs *args, char *mode) {
    if (strcmp(mode, "auto") != 0 && strcmp(mode, "always") != 0 && strcmp(mode, "never") != 0)
        return fail("--color must be one of: auto, always, never");
    args->output.color_mode = tn_color_mode_parse(mode);
    return TN_OK;
}

/* The value is stored before it is checked, as the flag has always done. */
static TernaryError set_update_repo(CliArgs *args, char *value) {
    args->update.update_repo = value;
    return update_repo_valid(value) ? TN_OK : fail("--update-repo must be owner/name");
}

static TernaryError set_update_api(CliArgs *args, char *value) {
    args->update.update_api = value;
    return update_api_base_valid(value) ? TN_OK : fail("--update-api must be an https:// URL");
}

static TernaryError set_update_channel(CliArgs *args, char *value) {
    args->update.update_channel = update_channel_parse(value);
    if (args->update.update_channel >= 0) return TN_OK;
    return fail("--update-channel must be auto, stable or prerelease");
}

typedef struct {
    const char *flag;
    TernaryError (*apply)(CliArgs *args, char *value);
} ValueFlag;

/* Flags whose value needs more than a range check. */
static const ValueFlag k_value_flags[] = {
    {"--color", set_color},
    {"--update-repo", set_update_repo},
    {"--update-api", set_update_api},
    {"--update-channel", set_update_channel},
};

TernaryError args_parse_extras(CliArgs *args, int argc, char **argv, int *ip, int *handled) {
    int hit          = 0;
    TernaryError err = args_table_parse(k_flags, sizeof(k_flags) / sizeof(k_flags[0]),
                                        "args_flags_extras", args, argc, argv, ip, &hit);
    if (err != TN_OK || hit) {
        if (hit) *handled = 1;
        return err;
    }
    int i = *ip;
    for (size_t k = 0; k < sizeof(k_value_flags) / sizeof(k_value_flags[0]); k++) {
        if (strcmp(argv[i], k_value_flags[k].flag) != 0 || i + 1 >= argc) continue;
        err = k_value_flags[k].apply(args, argv[i + 1]);
        if (err != TN_OK) return err;
        *ip      = i + 1;
        *handled = 1;
        return TN_OK;
    }
    return TN_OK; /* not one of this group's flags */
}
