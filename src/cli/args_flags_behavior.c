/*
 * src/cli/args_flags_behavior.c
 *
 * Command-line flags: chat format, web UI branding, memory DB, calibration, numerics, output switches and the HTTP server. One group of parse_args() (args.c); each branch is unchanged, and
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

TernaryError args_parse_behavior(CliArgs *args, int argc, char **argv, int *ip, int *handled) {
    int i = *ip;
    if (strcmp(argv[i], "--chat-format") == 0 && i + 1 < argc) {
        args->chat_format = chat_template_parse(argv[++i]);
        if (args->chat_format < -1) {
            fprintf(stderr, "Error: --chat-format must be auto, chatml, llama3, deepseek, "
                            "mistral or plain\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--ui-title") == 0 && i + 1 < argc) {
        args->ui_title = argv[++i];
    } else if (strcmp(argv[i], "--ui-logo") == 0 && i + 1 < argc) {
        const char *m = argv[++i];
        if (strcmp(m, "ascii") == 0)
            args->ui_logo = 0;
        else if (strcmp(m, "text") == 0)
            args->ui_logo = 1;
        else {
            fprintf(stderr, "Error: --ui-logo must be ascii or text\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--watermark-detect") == 0 && i + 1 < argc) {
        args->watermark_detect = argv[++i];
    } else if (strcmp(argv[i], "--memory-db") == 0 && i + 1 < argc) {
        args->memory_db_path = argv[++i];
    } else if (strcmp(argv[i], "--classifier") == 0 && i + 1 < argc) {
        const char *fmt = argv[++i];
        if (strcmp(fmt, "auto") == 0)
            args->classifier_override = -1;
        else if (strcmp(fmt, "bf16") == 0)
            args->classifier_override = 0;
        else if (strcmp(fmt, "int8") == 0)
            args->classifier_override = 1;
        else if (strcmp(fmt, "int4") == 0)
            args->classifier_override = 2;
        else if (strcmp(fmt, "auto-fast") == 0)
            args->classifier_override = 3;
        else {
            fprintf(stderr,
                    "Error: --classifier must be one of: auto, bf16, int8, int4, auto-fast\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--simd") == 0 && i + 1 < argc) {
        const char *backend = argv[++i];
        if (strcmp(backend, "auto") == 0) {
            args->simd_override = NULL;
        } else if (strcmp(backend, "avx2") == 0 || strcmp(backend, "avx512f") == 0 ||
                   strcmp(backend, "vnni") == 0 || strcmp(backend, "scalar") == 0) {
            args->simd_override = (char *)backend;
        } else {
            fprintf(stderr, "Error: --simd must be one of: auto, avx2, avx512f, vnni, scalar\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--calibrate") == 0) {
        args->calibrate = true;
    } else if (strcmp(argv[i], "--no-calibrate") == 0) {
        args->no_calibrate = true;
    } else if (strcmp(argv[i], "--act-quant") == 0 && i + 1 < argc) {
        TnActQuant m;
        if (tn_act_quant_parse(argv[++i], &m) != 0) {
            fprintf(stderr, "Error: --act-quant must be one of: auto, int8, float\n");
            return TN_ERR_INVALID_CONFIG;
        }
        args->act_quant = (int)m;
    } else if (strcmp(argv[i], "--quality-tolerance") == 0 && i + 1 < argc) {
        ARG_NUM(args->quality_tolerance);
        if (!(args->quality_tolerance >= 0.0 && args->quality_tolerance <= 100.0)) {
            fprintf(stderr, "Error: --quality-tolerance must be 0..100 (percent)\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--self-check-tokens") == 0 && i + 1 < argc) {
        ARG_INT(args->self_check_tokens);
        if (args->self_check_tokens < 16 || args->self_check_tokens > 4096) {
            fprintf(stderr, "Error: --self-check-tokens must be 16..4096\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--self-check") == 0) {
        args->self_check = true;
    } else if (strcmp(argv[i], "--reasoning") == 0) {
        args->enable_reasoning = true;
    } else if (strcmp(argv[i], "--json") == 0) {
        args->json_mode = true;
    } else if (strcmp(argv[i], "--raw-prompt") == 0) {
        args->raw_prompt = true;
    } else if (strcmp(argv[i], "--ignore-eos") == 0) {
        args->ignore_eos = true;
    } else if (strcmp(argv[i], "--verbose") == 0) {
        args->verbose = true;
    } else if (strcmp(argv[i], "--dump-tensors") == 0 && i + 1 < argc) {
        args->dump_tensors_path = argv[++i];
    } else if (strcmp(argv[i], "--server") == 0) {
        args->server_mode = 1;
    } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
        ARG_INT(args->server_port);
        if (args->server_port <= 0 || args->server_port > 65535) {
            fprintf(stderr, "Error: --port must be 1–65535\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else if (strcmp(argv[i], "--cors") == 0) {
        args->cors_enabled = true;
    } else if (strcmp(argv[i], "--cors-origin") == 0 && i + 1 < argc) {
        if (args->num_cors_origins >= CLI_MAX_CORS_ORIGINS) {
            fprintf(stderr, "Error: too many --cors-origin values (max %d)\n",
                    CLI_MAX_CORS_ORIGINS);
            return TN_ERR_INVALID_CONFIG;
        }
        args->cors_origins[args->num_cors_origins++] = argv[++i];
    } else if (strcmp(argv[i], "--api-key") == 0 && i + 1 < argc) {
        args->api_key = argv[++i];
    } else if (strcmp(argv[i], "--metrics") == 0) {
        args->metrics_enabled = true;
    } else if (strcmp(argv[i], "--static-dir") == 0 && i + 1 < argc) {
        args->static_dir = argv[++i];
    } else if (strcmp(argv[i], "--web-ui") == 0 && i + 1 < argc) {
        const char *mode = argv[++i];
        if (strcmp(mode, "auto") == 0)
            args->web_ui_mode = WEBUI_MODE_AUTO;
        else if (strcmp(mode, "on") == 0)
            args->web_ui_mode = WEBUI_MODE_ON;
        else if (strcmp(mode, "off") == 0)
            args->web_ui_mode = WEBUI_MODE_OFF;
        else {
            fprintf(stderr, "Error: --web-ui must be one of: auto, on, off\n");
            return TN_ERR_INVALID_CONFIG;
        }
    } else {
        return TN_OK; /* not one of this group's flags */
    }
    *ip      = i;
    *handled = 1;
    return TN_OK;
}
