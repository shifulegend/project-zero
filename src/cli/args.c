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

static void defaults_input(CliInput *g) {
    g->model_path         = NULL;
    g->tokenizer_path     = NULL;
    g->prompt             = NULL;
    g->image_path         = NULL;
    g->vision_path        = NULL;
    g->proj_path          = NULL;
    g->memory_db_path     = NULL;
    g->draft_model_path   = NULL; /* disabled by default -- the ONLY way to enable
                                       speculative decoding is --draft-model */
    g->lora_path          = NULL; /* disabled by default -- the ONLY way to enable a
                                LoRA adapter is --lora */
    g->system_prompt      = NULL;
    g->chat_template_file = NULL;
}

static void defaults_sampling(CliSampling *g) {
    g->temperature      = 0.7f;
    g->top_p            = 0.9f;
    g->max_tokens       = 512;
    g->json_mode        = false;
    g->raw_prompt       = false;
    g->ignore_eos       = false;
    g->seed             = -1;
    g->n_stop           = 0;
    g->max_tokens_limit = 0;
    g->chat_format      = -1;
    g->best_of          = 1;
    g->best_of_max      = BEST_OF_MAX_DEFAULT;
    {
        SamplerSettings ss;
        sampler_settings_default(&ss);
        g->top_k             = ss.top_k;
        g->min_p             = ss.min_p;
        g->repeat_penalty    = ss.repetition_penalty;
        g->presence_penalty  = ss.presence_penalty;
        g->frequency_penalty = ss.frequency_penalty;
        g->repeat_last_n     = ss.penalty_last_n;
    }
}

static void defaults_eval(CliEval *g) {
    g->perplexity_file   = NULL;
    g->ppl_ctx           = 512;
    g->ppl_chunks        = 0;
    g->act_quant         = 0;
    g->quality_tolerance = 1.0;
    g->self_check_tokens = 96;
    g->self_check        = false;
    g->dump_tensors_path = NULL;
}

static void defaults_perf(CliPerf *g) {
    g->num_threads             = 0; // auto
    g->rope_layout             = -1;
    g->classifier_override     = -1;   // auto (BF16)
    g->simd_override           = NULL; // auto
    g->calibrate               = false;
    g->no_calibrate            = false;
    g->prompt_cache            = true;
    g->prompt_cache_mb         = 256;
    g->prompt_cache_min_tokens = 32;
    g->ctx_size                = 0;
    g->kv_type                 = 0;
    g->kv_offload_dir          = NULL;
    g->attn_kernel             = NULL;
    g->attn_tile               = 0;
    g->attn_parallel_min_ctx   = ATTN_PARALLEL_MIN_CTX_DEFAULT;
    g->kv_ram_fraction         = 0.0f;
}

static void defaults_update(CliUpdate *g) {
    g->check_update      = false;
    g->no_update_check   = false;
    g->update_repo       = NULL;
    g->update_api        = NULL;
    g->update_interval_h = -1;
    g->update_channel    = UPDATE_CHANNEL_AUTO;
}

static void defaults_server(CliServer *g) {
    g->server_mode         = 0;
    g->server_port         = 8080;
    g->cors_enabled        = false;
    g->num_cors_origins    = 0;
    g->api_key             = NULL;
    g->metrics_enabled     = false;
    g->static_dir          = NULL;
    g->web_ui_mode         = WEBUI_MODE_AUTO;
    g->no_request_template = false;
    g->ui_title            = NULL;
    g->ui_logo             = 0;
    g->queue_size          = 0;
    g->queue_timeout_s     = 120;
    g->rate_limit          = 20;
    g->client_timeout_s    = 30;
    g->max_connections     = 64;
    g->max_prompt_bytes    = 1024 * 1024;
    g->parallel            = 1;
    g->parallel_batch      = PZ_PARALLEL_BATCH_DEFAULT;
}

static void defaults_output(CliOutput *g) {
    g->verbose      = false;
    g->show_version = false;
    g->color_mode   = TN_COLOR_AUTO;
    g->log_json     = 0;
    g->log_level    = -1;
}

static void defaults_search(CliSearch *g) {
    g->enable_reasoning = false;
    g->spec_length      = 5;
    g->mcts_max         = 64;
    g->prm_path = g->prm_format = g->prm_good = g->prm_bad = NULL;
    memset(&g->mcts, 0, sizeof(g->mcts));
}

static void defaults_watermark(CliWatermark *g) {
    g->enabled          = false;
    g->watermark_key    = NULL;
    g->watermark_gamma  = WATERMARK_DEFAULT_GAMMA;
    g->watermark_delta  = WATERMARK_DEFAULT_DELTA;
    g->watermark_detect = NULL;
}

static void defaults_dist(CliDist *g) {
    g->rpc_worker_port = 0;
    g->worker_layers   = NULL;
    g->rpc_bind_all    = false;
    g->remote_layers   = NULL;
    g->plan_layers     = NULL;
}

/* The value of every option before a flag changes it. */
static void args_set_defaults(CliArgs *args) {
    defaults_input(&args->input);
    defaults_sampling(&args->sampling);
    defaults_eval(&args->eval);
    defaults_perf(&args->perf);
    defaults_update(&args->update);
    defaults_server(&args->server);
    defaults_output(&args->output);
    defaults_search(&args->search);
    defaults_watermark(&args->watermark);
    defaults_dist(&args->dist);
}

TernaryError parse_args(CliArgs *args, int argc, char **argv) {
    if (!args) return TN_ERR_INVALID_CONFIG;

    args_set_defaults(args);

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
            args->output.show_version = true;
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

    if (args->perf.calibrate && args->perf.no_calibrate) {
        TN_LOG_ERR("args",
                   "Error: --calibrate and --no-calibrate contradict each other; use one.\n");
        return TN_ERR_INVALID_CONFIG;
    }

    if (args->update.check_update) return TN_OK; /* needs no model */

    if (!args->input.model_path) {
        TN_LOG_ERR("args", "Error: --model <path> is required.\n");
        print_usage(argv[0]);
        return TN_ERR_INVALID_CONFIG;
    }

    /* Only warn about missing tokenizer for non-GGUF models.
     * GGUF files embed tokenizer metadata (vocab, BPE merges, chat template,
     * BOS/EOS token IDs) and are loaded automatically via tokenizer_load_from_gguf(). */
    if (!args->input.tokenizer_path) {
        const char *mp = args->input.model_path ? args->input.model_path : "";
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
    gp->max_tokens                 = args->sampling.max_tokens;
    gp->sampler.temperature        = args->sampling.temperature;
    gp->sampler.top_p              = args->sampling.top_p;
    gp->sampler.top_k              = args->sampling.top_k;
    gp->sampler.min_p              = args->sampling.min_p;
    gp->sampler.repetition_penalty = args->sampling.repeat_penalty;
    gp->sampler.presence_penalty   = args->sampling.presence_penalty;
    gp->sampler.frequency_penalty  = args->sampling.frequency_penalty;
    gp->sampler.penalty_last_n     = args->sampling.repeat_last_n;
    sampler_settings_sanitize(&gp->sampler);
    gp->seed          = args->sampling.seed;
    gp->json_mode     = args->sampling.json_mode;
    gp->raw_prompt    = args->sampling.raw_prompt;
    gp->ignore_eos    = args->sampling.ignore_eos;
    gp->system_prompt = args->input.system_prompt;
    for (int i = 0; i < args->sampling.n_stop && i < GEN_MAX_STOP; i++)
        gp->stop[i] = args->sampling.stop[i];
    gp->n_stop = args->sampling.n_stop < GEN_MAX_STOP ? args->sampling.n_stop : GEN_MAX_STOP;
    gp->watermark.enabled = args->watermark.enabled;
    gp->watermark.key     = watermark_key_from_string(args->watermark.watermark_key);
    gp->watermark.gamma   = args->watermark.watermark_gamma;
    gp->watermark.delta   = args->watermark.watermark_delta;
    watermark_config_sanitize(&gp->watermark);
}

void args_fill_update_options(const CliArgs *args, UpdateOptions *o) {
    update_options_default(o);
    if (args->update.no_update_check) o->enabled = 0;
    if (args->update.update_interval_h >= 0) o->interval_hours = args->update.update_interval_h;
    o->channel = args->update.update_channel;
    if (args->update.update_repo)
        snprintf(o->repo, sizeof(o->repo), "%s", args->update.update_repo);
    if (args->update.update_api)
        snprintf(o->api_base, sizeof(o->api_base), "%s", args->update.update_api);
}
