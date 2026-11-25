/*
 * src/cli/args_flags_behavior.c
 *
 * Command-line flags: chat format, web UI branding, memory DB, calibration, numerics, output
 * switches and the HTTP server. One group of parse_args() (args.c); each branch is unchanged, and
 * tests/test_args_golden.c compares every flag's parse result with a golden table.
 */

#include <stddef.h>
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

/* ── Flags without a value: set a field ──────────────────────────────────── */

typedef struct {
    const char *flag;
    size_t offset; /* of the field inside CliArgs */
    int wide;      /* 1: the field is an int (set to 1); 0: a bool (set to true) */
} SwitchFlag;

static const SwitchFlag k_switch_flags[] = {
    {"--calibrate", offsetof(CliArgs, perf.calibrate), 0},
    {"--no-calibrate", offsetof(CliArgs, perf.no_calibrate), 0},
    {"--self-check", offsetof(CliArgs, eval.self_check), 0},
    {"--reasoning", offsetof(CliArgs, search.enable_reasoning), 0},
    {"--json", offsetof(CliArgs, sampling.json_mode), 0},
    {"--raw-prompt", offsetof(CliArgs, sampling.raw_prompt), 0},
    {"--ignore-eos", offsetof(CliArgs, sampling.ignore_eos), 0},
    {"--verbose", offsetof(CliArgs, output.verbose), 0},
    {"--server", offsetof(CliArgs, server.server_mode), 1},
    {"--cors", offsetof(CliArgs, server.cors_enabled), 0},
    {"--metrics", offsetof(CliArgs, server.metrics_enabled), 0},
};

static int parse_switch(CliArgs *args, const char *flag) {
    for (size_t k = 0; k < sizeof(k_switch_flags) / sizeof(k_switch_flags[0]); k++) {
        if (strcmp(flag, k_switch_flags[k].flag) != 0) continue;
        char *dst = (char *)args + k_switch_flags[k].offset;
        if (k_switch_flags[k].wide)
            *(int *)dst = 1;
        else
            *(bool *)dst = true;
        return 1;
    }
    return 0;
}

/* ── Flags whose value is kept as given (a pointer into argv) ────────────── */

static const struct {
    const char *flag;
    size_t offset; /* of a char * inside CliArgs */
} k_string_flags[] = {
    {"--ui-title", offsetof(CliArgs, server.ui_title)},
    {"--watermark-detect", offsetof(CliArgs, watermark.watermark_detect)},
    {"--memory-db", offsetof(CliArgs, input.memory_db_path)},
    {"--dump-tensors", offsetof(CliArgs, eval.dump_tensors_path)},
    {"--api-key", offsetof(CliArgs, server.api_key)},
    {"--static-dir", offsetof(CliArgs, server.static_dir)},
};

typedef char webui_mode_is_int_sized[sizeof(WebUiMode) == sizeof(int) ? 1 : -1];

/* ── Flags that pick one of a few named values ───────────────────────────── */

typedef struct {
    const char *name;
    int value;
} Choice;

static const Choice k_ui_logo[]    = {{"ascii", 0}, {"text", 1}};
static const Choice k_classifier[] = {
    {"auto", -1}, {"bf16", 0}, {"int8", 1}, {"int4", 2}, {"auto-fast", 3}};
static const Choice k_web_ui[] = {
    {"auto", WEBUI_MODE_AUTO}, {"on", WEBUI_MODE_ON}, {"off", WEBUI_MODE_OFF}};

static const struct {
    const char *flag;
    size_t offset; /* of an int-sized field inside CliArgs */
    const Choice *choices;
    size_t n_choices;
    const char *message; /* printed (after "Error: ") when the value is none of them */
} k_choice_flags[] = {
    {"--ui-logo", offsetof(CliArgs, server.ui_logo), k_ui_logo, 2,
     "--ui-logo must be ascii or text"},
    {"--classifier", offsetof(CliArgs, perf.classifier_override), k_classifier, 5,
     "--classifier must be one of: auto, bf16, int8, int4, auto-fast"},
    {"--web-ui", offsetof(CliArgs, server.web_ui_mode), k_web_ui, 3,
     "--web-ui must be one of: auto, on, off"},
};

/* ── Numeric flags with a range ──────────────────────────────────────────── */

static const struct {
    const char *flag;
    size_t offset;
    int is_double; /* 1: a double field; 0: an int field */
    double lo, hi;
    const char *message;
} k_range_flags[] = {
    {"--quality-tolerance", offsetof(CliArgs, eval.quality_tolerance), 1, 0.0, 100.0,
     "--quality-tolerance must be 0..100 (percent)"},
    {"--self-check-tokens", offsetof(CliArgs, eval.self_check_tokens), 0, 16, 4096,
     "--self-check-tokens must be 16..4096"},
    {"--port", offsetof(CliArgs, server.server_port), 0, 1, 65535,
     "--port must be 1\xE2\x80\x93"
     "65535"},
};

/* ── Flags with their own value rules ────────────────────────────────────── */

static TernaryError parse_chat_format(CliArgs *args, const char *value) {
    args->sampling.chat_format = chat_template_parse(value);
    if (args->sampling.chat_format >= -1) return TN_OK;
    TN_LOG_ERR("args_flags_behavior",
               "Error: --chat-format must be auto, chatml, llama3, deepseek, "
               "mistral or plain\n");
    return TN_ERR_INVALID_CONFIG;
}

static TernaryError parse_simd(CliArgs *args, const char *backend) {
    if (strcmp(backend, "auto") == 0) {
        args->perf.simd_override = NULL;
    } else if (strcmp(backend, "avx2") == 0 || strcmp(backend, "avx512f") == 0 ||
               strcmp(backend, "vnni") == 0 || strcmp(backend, "scalar") == 0) {
        args->perf.simd_override = (char *)backend;
    } else {
        TN_LOG_ERR("args_flags_behavior",
                   "Error: --simd must be one of: auto, avx2, avx512f, vnni, scalar\n");
        return TN_ERR_INVALID_CONFIG;
    }
    return TN_OK;
}

static TernaryError parse_act_quant(CliArgs *args, const char *value) {
    TnActQuant m;
    if (tn_act_quant_parse(value, &m) != 0) {
        TN_LOG_ERR("args_flags_behavior", "Error: --act-quant must be one of: auto, int8, float\n");
        return TN_ERR_INVALID_CONFIG;
    }
    args->eval.act_quant = (int)m;
    return TN_OK;
}

static TernaryError parse_log_format(CliArgs *args, const char *value) {
    TnLogFormat f;
    if (tn_log_parse_format(value, &f) != 0) {
        TN_LOG_ERR("args_flags_behavior", "Error: --log-format must be one of: text, json\n");
        return TN_ERR_INVALID_CONFIG;
    }
    args->output.log_json = f == TN_LOG_JSON;
    return TN_OK;
}

static TernaryError parse_log_level(CliArgs *args, const char *value) {
    TnLogLevel l;
    if (tn_log_parse_level(value, &l) != 0) {
        TN_LOG_ERR("args_flags_behavior",
                   "Error: --log-level must be one of: debug, info, warn, error\n");
        return TN_ERR_INVALID_CONFIG;
    }
    args->output.log_level = (int)l;
    return TN_OK;
}

static TernaryError parse_cors_origin(CliArgs *args, const char *value) {
    if (args->server.num_cors_origins >= CLI_MAX_CORS_ORIGINS) {
        TN_LOG_ERR("args_flags_behavior", "Error: too many --cors-origin values (max %d)\n",
                   CLI_MAX_CORS_ORIGINS);
        return TN_ERR_INVALID_CONFIG;
    }
    args->server.cors_origins[args->server.num_cors_origins++] = (char *)value;
    return TN_OK;
}

static const struct {
    const char *flag;
    TernaryError (*parse)(CliArgs *args, const char *value);
} k_handler_flags[] = {
    {"--chat-format", parse_chat_format}, {"--simd", parse_simd},
    {"--act-quant", parse_act_quant},     {"--log-format", parse_log_format},
    {"--log-level", parse_log_level},     {"--cors-origin", parse_cors_origin},
};

/* Each of these handles one kind of valued flag, with the value known to be at argv[*ip + 1]: it
 * sets *hit and advances *ip past the value when argv[*ip] is one of its flags. */

static void parse_string_flag(CliArgs *args, char **argv, int *ip, int *hit) {
    for (size_t k = 0; k < sizeof(k_string_flags) / sizeof(k_string_flags[0]); k++) {
        if (strcmp(argv[*ip], k_string_flags[k].flag) != 0) continue;
        *(char **)((char *)args + k_string_flags[k].offset) = argv[*ip + 1];
        *ip += 1;
        *hit = 1;
        return;
    }
}

static TernaryError parse_choice_flag(CliArgs *args, char **argv, int *ip, int *hit) {
    for (size_t k = 0; k < sizeof(k_choice_flags) / sizeof(k_choice_flags[0]); k++) {
        if (strcmp(argv[*ip], k_choice_flags[k].flag) != 0) continue;
        *hit = 1;
        for (size_t c = 0; c < k_choice_flags[k].n_choices; c++)
            if (strcmp(argv[*ip + 1], k_choice_flags[k].choices[c].name) == 0) {
                *(int *)((char *)args + k_choice_flags[k].offset) =
                    k_choice_flags[k].choices[c].value;
                *ip += 1;
                return TN_OK;
            }
        TN_LOG_ERR("args_flags_behavior", "Error: %s\n", k_choice_flags[k].message);
        return TN_ERR_INVALID_CONFIG;
    }
    return TN_OK;
}

static TernaryError parse_range_flag(CliArgs *args, char **argv, int *ip, int *hit) {
    int i = *ip;
    for (size_t k = 0; k < sizeof(k_range_flags) / sizeof(k_range_flags[0]); k++) {
        if (strcmp(argv[i], k_range_flags[k].flag) != 0) continue;
        *hit      = 1;
        char *dst = (char *)args + k_range_flags[k].offset;
        double v;
        if (k_range_flags[k].is_double) {
            ARG_NUM(*(double *)dst);
            v = *(double *)dst;
        } else {
            ARG_INT(*(int *)dst);
            v = *(int *)dst;
        }
        if (!(v >= k_range_flags[k].lo && v <= k_range_flags[k].hi)) {
            TN_LOG_ERR("args_flags_behavior", "Error: %s\n", k_range_flags[k].message);
            return TN_ERR_INVALID_CONFIG;
        }
        *ip = i;
        return TN_OK;
    }
    return TN_OK;
}

static TernaryError parse_handler_flag(CliArgs *args, char **argv, int *ip, int *hit) {
    for (size_t k = 0; k < sizeof(k_handler_flags) / sizeof(k_handler_flags[0]); k++) {
        if (strcmp(argv[*ip], k_handler_flags[k].flag) != 0) continue;
        *hit             = 1;
        TernaryError err = k_handler_flags[k].parse(args, argv[*ip + 1]);
        if (err == TN_OK) *ip += 1;
        return err;
    }
    return TN_OK;
}

/* The flags that take a value, argv[*ip] already known to have one after it. */
static TernaryError parse_valued(CliArgs *args, char **argv, int *ip, int *hit) {
    parse_string_flag(args, argv, ip, hit);
    if (*hit) return TN_OK;
    TernaryError err = parse_choice_flag(args, argv, ip, hit);
    if (*hit) return err;
    err = parse_range_flag(args, argv, ip, hit);
    if (*hit) return err;
    return parse_handler_flag(args, argv, ip, hit);
}

TernaryError args_parse_behavior(CliArgs *args, int argc, char **argv, int *ip, int *handled) {
    int i            = *ip;
    int hit          = parse_switch(args, argv[i]);
    TernaryError err = TN_OK;
    if (!hit && i + 1 < argc) err = parse_valued(args, argv, &i, &hit);
    if (err != TN_OK || !hit) return err;
    *ip      = i;
    *handled = 1;
    return TN_OK;
}
