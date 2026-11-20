#include "cli/args.h"

#include <stdio.h>
#include <string.h>

#include "api/chat_compile.h"
#include "api/server_config.h"
#include "cli/arg_values.h"
#include "cli/args_groups.h"
#include "core/log.h"
#include "kv_cache/kv_quant_ops.h"
#include "math/rope.h"
#include "math/simd_dispatch.h"
#include "reasoning/best_of.h"
#include "transformer/attn_tiled.h"
#include "update/update_check.h"

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
    args->log_json            = 0;
    args->log_level           = -1;
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
        int handled = 0;
        TernaryError e;
        if ((e = args_parse_generation(args, argc, argv, &i, &handled)) != TN_OK) return e;
        if (!handled && (e = args_parse_memory(args, argc, argv, &i, &handled)) != TN_OK) return e;
        if (!handled && (e = args_parse_serving(args, argc, argv, &i, &handled)) != TN_OK) return e;
        if (!handled && (e = args_parse_search(args, argc, argv, &i, &handled)) != TN_OK) return e;
        if (!handled && (e = args_parse_behavior(args, argc, argv, &i, &handled)) != TN_OK)
            return e;
        if (!handled && (e = args_parse_extras(args, argc, argv, &i, &handled)) != TN_OK) return e;
        if (handled) continue;
        if (strcmp(argv[i], "--version") == 0 || strcmp(argv[i], "-v") == 0) {
            /* Short-circuit: --version must work without --model so the release
             * binary is smoke-testable on a clean machine with no model file.
             * main() prints the version and exits 0 before any model load. */
            args->show_version = true;
            return TN_OK;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return TN_ERR_INVALID_CONFIG;
        } else {
            TN_LOG_ERR("args", "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return TN_ERR_INVALID_CONFIG;
        }
    }

    if (args->calibrate && args->no_calibrate) {
        TN_LOG_ERR("args",
                   "Error: --calibrate and --no-calibrate contradict each other; use one.\n");
        return TN_ERR_INVALID_CONFIG;
    }

    if (args->check_update) return TN_OK; /* needs no model */

    if (!args->model_path) {
        TN_LOG_ERR("args", "Error: --model <path> is required.\n");
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
            TN_LOG_WRN("args", "Warning: --tokenizer <path> not provided. "
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
