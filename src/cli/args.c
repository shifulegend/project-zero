#include "cli/args.h"
#include "cli/arg_values.h"
#include "math/simd_dispatch.h"
#include "transformer/attn_tiled.h"
#include "kv_cache/kv_quant_ops.h"
#include "api/chat_compile.h"
#include "reasoning/best_of.h"
#include "api/server_config.h"
#include "update/update_check.h"
#include "math/rope.h"
#include <stdio.h>
#include <string.h>

TernaryError parse_args(CliArgs *args, int argc, char **argv) {
    if (!args) return TN_ERR_INVALID_CONFIG;

    // Set defaults
    args->model_path          = NULL;
    args->tokenizer_path      = NULL;
    args->prompt              = NULL;
    args->image_path          = NULL;
    args->vision_path         = NULL;
    args->proj_path           = NULL;
    args->memory_db_path      = NULL;
    args->temperature         = 0.7f;
    args->top_p               = 0.9f;
    args->max_tokens          = 512;
    args->num_threads         = 0; // auto
    args->enable_reasoning    = false;
    args->json_mode           = false;
    args->raw_prompt          = false;
    args->ignore_eos          = false;
    args->rope_layout         = -1;
    args->perplexity_file     = NULL;
    args->ppl_ctx             = 512;
    args->ppl_chunks          = 0;
    args->verbose             = false;
    args->seed                = -1;
    args->classifier_override = -1;   // auto (BF16)
    args->simd_override       = NULL; // auto
    args->calibrate           = false;
    args->no_calibrate        = false;
    args->act_quant           = 0;
    args->quality_tolerance   = 1.0;
    args->self_check_tokens   = 96;
    args->self_check          = false;
    args->show_version        = false;
    args->check_update        = false;
    args->no_update_check     = false;
    args->update_repo         = NULL;
    args->update_api          = NULL;
    args->update_interval_h   = -1;
    args->update_channel      = UPDATE_CHANNEL_AUTO;
    args->dump_tensors_path   = NULL;
    args->server_mode         = 0;
    args->server_port         = 8080;
    args->cors_enabled        = false;
    args->num_cors_origins    = 0;
    args->api_key             = NULL;
    args->metrics_enabled     = false;
    args->static_dir          = NULL;
    args->web_ui_mode         = WEBUI_MODE_AUTO;
    args->color_mode          = TN_COLOR_AUTO;
    args->draft_model_path    = NULL; /* disabled by default -- the ONLY way to enable
                                       speculative decoding is --draft-model */
    args->spec_length         = 5;
    args->lora_path           = NULL; /* disabled by default -- the ONLY way to enable a
                                LoRA adapter is --lora */
    {
        SamplerSettings ss;
        sampler_settings_default(&ss);
        args->top_k             = ss.top_k;
        args->min_p             = ss.min_p;
        args->repeat_penalty    = ss.repetition_penalty;
        args->presence_penalty  = ss.presence_penalty;
        args->frequency_penalty = ss.frequency_penalty;
        args->repeat_last_n     = ss.penalty_last_n;
    }
    args->system_prompt       = NULL;
    args->chat_template_file  = NULL;
    args->no_request_template = false;
    args->n_stop              = 0;
    args->max_tokens_limit    = 0;
    args->watermark           = false;
    args->watermark_key       = NULL;
    args->watermark_gamma     = WATERMARK_DEFAULT_GAMMA;
    args->watermark_delta     = WATERMARK_DEFAULT_DELTA;
    args->watermark_detect    = NULL;
    args->ui_title            = NULL;
    args->ui_logo             = 0;
    args->chat_format         = -1;
    args->best_of             = 1;
    args->queue_size          = 0;
    args->rpc_worker_port     = 0;
    args->worker_layers       = NULL;
    args->rpc_bind_all        = false;
    args->remote_layers       = NULL;
    args->plan_layers         = NULL;
    args->queue_timeout_s     = 120;
    args->rate_limit          = 20;
    args->client_timeout_s    = 30;
    args->max_connections     = 64;
    args->parallel            = 1;
    args->parallel_batch      = PZ_PARALLEL_BATCH_DEFAULT;
    args->best_of_max         = BEST_OF_MAX_DEFAULT;
    memset(&args->mcts, 0, sizeof(args->mcts));
    args->mcts_max = 64;
    args->prm_path = args->prm_format = args->prm_good = args->prm_bad = NULL;
    args->prompt_cache                                                 = true;
    args->prompt_cache_mb                                              = 256;
    args->prompt_cache_min_tokens                                      = 32;
    args->ctx_size                                                     = 0;
    args->kv_type                                                      = 0;
    args->kv_offload_dir                                               = NULL;
    args->attn_kernel                                                  = NULL;
    args->attn_tile                                                    = 0;
    args->attn_parallel_min_ctx = ATTN_PARALLEL_MIN_CTX_DEFAULT;
    args->kv_ram_fraction       = 0.0f;

    for (int i = 1; i < argc; i++) {
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
                fprintf(stderr, "Error: too many --stop values (max %d)\n", GEN_MAX_STOP);
                return TN_ERR_INVALID_CONFIG;
            }
            args->stop[args->n_stop++] = argv[++i];
        } else if (strcmp(argv[i], "--max-tokens-limit") == 0 && i + 1 < argc) {
            ARG_INT(args->max_tokens_limit);
            if (args->max_tokens_limit < 0) {
                fprintf(stderr, "Error: --max-tokens-limit must be >= 0\n");
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
        } else if (strcmp(argv[i], "--perplexity") == 0 && i + 1 < argc) {
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
        } else if (strcmp(argv[i], "--rpc-worker") == 0 && i + 1 < argc) {
            ARG_INT(args->rpc_worker_port);
            if (args->rpc_worker_port <= 0 || args->rpc_worker_port > 65535) {
                fprintf(stderr, "Error: --rpc-worker must be a port 1-65535\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            args->worker_layers = argv[++i];
        } else if (strcmp(argv[i], "--rpc-bind-all") == 0) {
            args->rpc_bind_all = true;
        } else if (strcmp(argv[i], "--remote-layers") == 0 && i + 1 < argc) {
            args->remote_layers = argv[++i];
        } else if (strcmp(argv[i], "--plan-layers") == 0 && i + 1 < argc) {
            args->plan_layers = argv[++i];
        } else if (strcmp(argv[i], "--queue-size") == 0 && i + 1 < argc) {
            ARG_INT(args->queue_size);
            if (args->queue_size < 0) {
                fprintf(stderr, "Error: --queue-size must be >= 0\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--queue-timeout") == 0 && i + 1 < argc) {
            ARG_INT(args->queue_timeout_s);
            if (args->queue_timeout_s < 1) {
                fprintf(stderr, "Error: --queue-timeout must be >= 1 second\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--rate-limit") == 0 && i + 1 < argc) {
            ARG_INT(args->rate_limit);
            if (args->rate_limit < 0 || args->rate_limit > 100000) {
                fprintf(stderr, "Error: --rate-limit must be 0..100000 requests per second\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--client-timeout") == 0 && i + 1 < argc) {
            ARG_INT(args->client_timeout_s);
            if (args->client_timeout_s < 0 || args->client_timeout_s > 86400) {
                fprintf(stderr, "Error: --client-timeout must be 0..86400 seconds\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--max-connections") == 0 && i + 1 < argc) {
            ARG_INT(args->max_connections);
            if (args->max_connections < 0 || args->max_connections > 1048576) {
                fprintf(stderr, "Error: --max-connections must be 0..1048576\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--parallel") == 0 && i + 1 < argc) {
            ARG_INT(args->parallel);
            if (args->parallel < 1 || args->parallel > PZ_PARALLEL_MAX) {
                fprintf(stderr, "Error: --parallel must be 1..%d\n", PZ_PARALLEL_MAX);
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--parallel-batch") == 0 && i + 1 < argc) {
            ARG_INT(args->parallel_batch);
            if (args->parallel_batch < 1 || args->parallel_batch > PZ_PARALLEL_BATCH_MAX) {
                fprintf(stderr, "Error: --parallel-batch must be 1..%d\n", PZ_PARALLEL_BATCH_MAX);
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--best-of") == 0 && i + 1 < argc) {
            ARG_INT(args->best_of);
            if (args->best_of < 1 || args->best_of > BEST_OF_N_MAX) {
                fprintf(stderr, "Error: --best-of must be 1..%d\n", BEST_OF_N_MAX);
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--mcts") == 0 && i + 1 < argc) {
            ARG_INT(args->mcts.simulations);
            if (args->mcts.simulations < 0 || args->mcts.simulations > 4096) {
                fprintf(stderr, "Error: --mcts must be 0..4096\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--mcts-max") == 0 && i + 1 < argc) {
            ARG_INT(args->mcts_max);
            if (args->mcts_max < 0 || args->mcts_max > 4096) {
                fprintf(stderr, "Error: --mcts-max must be 0..4096\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--mcts-expand") == 0 && i + 1 < argc) {
            ARG_INT(args->mcts.expand);
            if (args->mcts.expand < 1 || args->mcts.expand > 16) {
                fprintf(stderr, "Error: --mcts-expand must be 1..16\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--mcts-depth") == 0 && i + 1 < argc) {
            ARG_INT(args->mcts.max_depth);
            if (args->mcts.max_depth < 1 || args->mcts.max_depth > 256) {
                fprintf(stderr, "Error: --mcts-depth must be 1..256\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--mcts-step-tokens") == 0 && i + 1 < argc) {
            ARG_INT(args->mcts.step_tokens);
            if (args->mcts.step_tokens < 1 || args->mcts.step_tokens > 4096) {
                fprintf(stderr, "Error: --mcts-step-tokens must be 1..4096\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--mcts-c") == 0 && i + 1 < argc) {
            ARG_NUM(args->mcts.c_ucb);
            if (!(args->mcts.c_ucb > 0.0f && args->mcts.c_ucb <= 10.0f)) {
                fprintf(stderr, "Error: --mcts-c must be in (0, 10]\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--mcts-temperature") == 0 && i + 1 < argc) {
            ARG_NUM(args->mcts.temperature);
            if (!(args->mcts.temperature > 0.0f && args->mcts.temperature <= 2.0f)) {
                fprintf(stderr, "Error: --mcts-temperature must be in (0, 2]\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--mcts-delim") == 0 && i + 1 < argc) {
            /* "\n" in the argument means a newline */
            const char *d = argv[++i];
            size_t k      = 0;
            for (; *d && k + 1 < sizeof(args->mcts_delim_buf); d++) {
                if (d[0] == '\\' && d[1] == 'n') {
                    args->mcts_delim_buf[k++] = '\n';
                    d++;
                } else
                    args->mcts_delim_buf[k++] = *d;
            }
            args->mcts_delim_buf[k] = '\0';
            if (k == 0) {
                fprintf(stderr, "Error: --mcts-delim must not be empty\n");
                return TN_ERR_INVALID_CONFIG;
            }
            args->mcts.step_delim = args->mcts_delim_buf;
        } else if (strcmp(argv[i], "--prm") == 0 && i + 1 < argc) {
            args->prm_path = argv[++i];
        } else if (strcmp(argv[i], "--prm-format") == 0 && i + 1 < argc) {
            args->prm_format = argv[++i];
        } else if (strcmp(argv[i], "--prm-good") == 0 && i + 1 < argc) {
            args->prm_good = argv[++i];
        } else if (strcmp(argv[i], "--prm-bad") == 0 && i + 1 < argc) {
            args->prm_bad = argv[++i];
        } else if (strcmp(argv[i], "--best-of-max") == 0 && i + 1 < argc) {
            ARG_INT(args->best_of_max);
            if (args->best_of_max < 1 || args->best_of_max > BEST_OF_N_MAX) {
                fprintf(stderr, "Error: --best-of-max must be 1..%d\n", BEST_OF_N_MAX);
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--chat-format") == 0 && i + 1 < argc) {
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
                fprintf(stderr,
                        "Error: --simd must be one of: auto, avx2, avx512f, vnni, scalar\n");
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
        } else if (strcmp(argv[i], "--draft-model") == 0 && i + 1 < argc) {
            args->draft_model_path = argv[++i];
        } else if (strcmp(argv[i], "--spec-length") == 0 && i + 1 < argc) {
            ARG_INT(args->spec_length);
            if (args->spec_length <= 0) {
                fprintf(stderr, "Error: --spec-length must be a positive integer\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--lora") == 0 && i + 1 < argc) {
            args->lora_path = argv[++i];
        } else if (strcmp(argv[i], "--color") == 0 && i + 1 < argc) {
            const char *mode = argv[++i];
            if (strcmp(mode, "auto") != 0 && strcmp(mode, "always") != 0 &&
                strcmp(mode, "never") != 0) {
                fprintf(stderr, "Error: --color must be one of: auto, always, never\n");
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
                fprintf(stderr, "Error: --update-repo must be owner/name\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--update-api") == 0 && i + 1 < argc) {
            args->update_api = argv[++i];
            if (!update_api_base_valid(args->update_api)) {
                fprintf(stderr, "Error: --update-api must be an https:// URL\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--update-interval") == 0 && i + 1 < argc) {
            ARG_INT(args->update_interval_h);
            if (args->update_interval_h < 0 || args->update_interval_h > 24 * 365) {
                fprintf(stderr, "Error: --update-interval must be 0-8760 hours\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--update-channel") == 0 && i + 1 < argc) {
            args->update_channel = update_channel_parse(argv[++i]);
            if (args->update_channel < 0) {
                fprintf(stderr, "Error: --update-channel must be auto, stable or prerelease\n");
                return TN_ERR_INVALID_CONFIG;
            }
        } else if (strcmp(argv[i], "--version") == 0 || strcmp(argv[i], "-v") == 0) {
            /* Short-circuit: --version must work without --model so the release
             * binary is smoke-testable on a clean machine with no model file.
             * main() prints the version and exits 0 before any model load. */
            args->show_version = true;
            return TN_OK;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return TN_ERR_INVALID_CONFIG;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return TN_ERR_INVALID_CONFIG;
        }
    }

    if (args->calibrate && args->no_calibrate) {
        fprintf(stderr, "Error: --calibrate and --no-calibrate contradict each other; use one.\n");
        return TN_ERR_INVALID_CONFIG;
    }

    if (args->check_update) return TN_OK; /* needs no model */

    if (!args->model_path) {
        fprintf(stderr, "Error: --model <path> is required.\n");
        print_usage(argv[0]);
        return TN_ERR_INVALID_CONFIG;
    }

    /* Only warn about missing tokenizer for non-GGUF models.
     * GGUF files embed tokenizer metadata (vocab, BPE merges, chat template,
     * BOS/EOS token IDs) and are loaded automatically via tokenizer_load_from_gguf(). */
    if (!args->tokenizer_path) {
        const char *mp = args->model_path ? args->model_path : "";
        int is_gguf    = (strstr(mp, ".gguf") != NULL);
        if (!is_gguf) {
            fprintf(stderr, "Warning: --tokenizer <path> not provided. "
                            "Text generation will fail without a vocabulary file.\n");
        }
    }

    return TN_OK;
}

void args_fill_gen_params(const CliArgs *args, GenerationParams *gp) {
    gen_params_default(gp);
    if (!args) return;
    gp->max_tokens                 = args->max_tokens;
    gp->sampler.temperature        = args->temperature;
    gp->sampler.top_p              = args->top_p;
    gp->sampler.top_k              = args->top_k;
    gp->sampler.min_p              = args->min_p;
    gp->sampler.repetition_penalty = args->repeat_penalty;
    gp->sampler.presence_penalty   = args->presence_penalty;
    gp->sampler.frequency_penalty  = args->frequency_penalty;
    gp->sampler.penalty_last_n     = args->repeat_last_n;
    sampler_settings_sanitize(&gp->sampler);
    gp->seed          = args->seed;
    gp->json_mode     = args->json_mode;
    gp->raw_prompt    = args->raw_prompt;
    gp->ignore_eos    = args->ignore_eos;
    gp->system_prompt = args->system_prompt;
    for (int i = 0; i < args->n_stop && i < GEN_MAX_STOP; i++) gp->stop[i] = args->stop[i];
    gp->n_stop            = args->n_stop < GEN_MAX_STOP ? args->n_stop : GEN_MAX_STOP;
    gp->watermark.enabled = args->watermark;
    gp->watermark.key     = watermark_key_from_string(args->watermark_key);
    gp->watermark.gamma   = args->watermark_gamma;
    gp->watermark.delta   = args->watermark_delta;
    watermark_config_sanitize(&gp->watermark);
}

void args_fill_update_options(const CliArgs *args, UpdateOptions *o) {
    update_options_default(o);
    if (args->no_update_check) o->enabled = 0;
    if (args->update_interval_h >= 0) o->interval_hours = args->update_interval_h;
    o->channel = args->update_channel;
    if (args->update_repo) snprintf(o->repo, sizeof(o->repo), "%s", args->update_repo);
    if (args->update_api) snprintf(o->api_base, sizeof(o->api_base), "%s", args->update_api);
}
