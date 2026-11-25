/*
 * Phase 22.2 — src/multimodal/vision_pipeline.c
 * See include/multimodal/vision_pipeline.h for responsibility.
 *
 * This is a mechanical extraction of the inline vision block that used to
 * live in src/cli/main.c (Phase 34) — logic is unchanged, only reshaped
 * into a reusable function with parameters instead of CliArgs/locals, so
 * both the CLI (--image) and the HTTP API (image_url content parts) can
 * call it without duplicating ~150 lines of vision pipeline plumbing.
 */

#include "multimodal/vision_pipeline.h"

#include <stdio.h>
#include <string.h>

#include "core/log.h"
#include "memory/aligned_alloc.h"
#include "multimodal/image_load.h"
#include "multimodal/patch_extract.h"
#include "multimodal/vision_bridge.h"
#include "multimodal/vision_encoder.h"
#include "multimodal/vision_projector.h"
#include "multimodal/vision_weights_load.h"
#include "transformer/forward.h"

/* Buffers of one run; every stage fills its own and the caller frees them all (NULL is fine). */
typedef struct {
    float *pixels;
    float *patches;
    float *vis_emb;
    float *projected;
    int target_res;
    int num_patches;
    int out_tokens;
} VisionBuffers;

static void vision_buffers_free(VisionBuffers *b) {
    tn_aligned_free(b->projected);
    tn_aligned_free(b->vis_emb);
    tn_aligned_free(b->patches);
    tn_aligned_free(b->pixels);
}

static int patch_size_of(const VisionModel *vm) {
    return (int)(0.5f + __builtin_sqrtf((float)(vm->cfg.patch_dim / 3)));
}

/* Load the vision encoder and projector weights. */
static int load_vision_weights(VisionModel *vm, const char *vision_path, const char *proj_path) {
    if (vision_model_load_encoder(vm, vision_path) == TN_OK &&
        vision_model_load_projector(vm, proj_path) == TN_OK) {
        vision_model_print_info(vm);
        return 1;
    }
    TN_LOG_ERR("vision", "[vision] failed to load vision weights\n");
    return 0;
}

/* Load and normalize the image. */
static int stage_load_image(const VisionModel *vm, const char *image_path, VisionBuffers *b) {
    int img_w = 0, img_h = 0;
    b->target_res = (int)(0.5f + __builtin_sqrtf((float)vm->cfg.num_patches)) * patch_size_of(vm);
    if (b->target_res <= 0) b->target_res = 384;

    if (load_image(image_path, &b->pixels, &img_w, &img_h, b->target_res) != TN_OK) {
        TN_LOG_ERR("vision", "[vision] failed to load image: %s\n", image_path);
        return 0;
    }
    printf("  Image loaded: %dx%d (SigLIP-normalized)\n", img_w, img_h);
    return 1;
}

/* Extract patches. */
static int stage_extract_patches(VisionModel *vm, VisionBuffers *b) {
    int patch_size = patch_size_of(vm);
    b->patches     = (float *)tn_aligned_alloc(
        (size_t)vm->cfg.num_patches * vm->cfg.patch_dim * sizeof(float), 64);
    if (!b->patches) {
        TN_LOG_ERR("vision", "[vision] OOM allocating patches\n");
        return 0;
    }
    extract_patches(b->pixels, b->patches, b->target_res, patch_size, &b->num_patches);
    printf("  Extracted %d patches (patch_size=%d)\n", b->num_patches, patch_size);
    vm->cfg.num_patches = b->num_patches;
    return 1;
}

/* Vision encoder forward. */
static int stage_encode(VisionModel *vm, ThreadPool *tp, VisionBuffers *b) {
    b->vis_emb =
        (float *)tn_aligned_alloc((size_t)b->num_patches * vm->cfg.embed_dim * sizeof(float), 64);
    if (!b->vis_emb) {
        TN_LOG_ERR("vision", "[vision] OOM: vision embeddings\n");
        return 0;
    }
    printf("  Running vision encoder (%d layers)...\n", vm->cfg.n_layers);
    vision_encoder_forward(b->vis_emb, b->patches, &vm->cfg, &vm->weights, tp);
    printf("  Encoder done.\n");
    return 1;
}

/* MLP projector (or pixel-shuffle + single linear), into LLM-dim embeddings: the projector's own
 * llm_dim is used for the projection, then copied (truncated or zero-padded) to the LLM's dim. */
static int stage_project(const VisionModel *vm, const Config *cfg, ThreadPool *tp,
                         VisionBuffers *b) {
    b->out_tokens = vision_projector_output_tokens(&vm->proj, b->num_patches);
    b->projected  = (float *)tn_aligned_alloc((size_t)b->out_tokens * cfg->dim * sizeof(float), 64);

    /* Projector llm_dim must match LLM dim; warn if mismatched */
    if (vm->proj.llm_dim != cfg->dim) {
        TN_LOG_WRN("vision",
                   "[vision] WARNING: projector llm_dim=%d != LLM dim=%d\n"
                   "[vision] Embeddings will be truncated/zero-padded.\n",
                   vm->proj.llm_dim, cfg->dim);
    }
    if (!b->projected) {
        TN_LOG_ERR("vision", "[vision] OOM: projected embeddings\n");
        return 0;
    }

    float *proj_buf =
        (float *)tn_aligned_alloc((size_t)b->out_tokens * vm->proj.llm_dim * sizeof(float), 64);
    if (!proj_buf) return 0;
    vision_projector_forward_batch(proj_buf, b->vis_emb, b->num_patches, &vm->proj, tp);
    /* Copy into projected, matching LLM dim */
    int copy_dim = vm->proj.llm_dim < cfg->dim ? vm->proj.llm_dim : cfg->dim;
    memset(b->projected, 0, (size_t)b->out_tokens * cfg->dim * sizeof(float));
    for (int pp = 0; pp < b->out_tokens; pp++) {
        memcpy(&b->projected[pp * cfg->dim], &proj_buf[pp * vm->proj.llm_dim],
               (size_t)copy_dim * sizeof(float));
    }
    tn_aligned_free(proj_buf);
    printf("  Projector done: %d tokens x %d-dim  (scale_factor=%d)\n", b->out_tokens, cfg->dim,
           vm->proj.scale_factor);
    return 1;
}

/* For ChatML-style models (chat_template contains "im_start"), pre-process a chat prefix so
 * visual tokens land at the correct sequence position (after "User: "), matching training
 * layout. Detected from the model's own chat_template — no vocab_size heuristics. */
static int wants_visual_prefix(const char *prompt_text, const Tokenizer *tok) {
    return prompt_text && tok->chat_template && strstr(tok->chat_template, "im_start");
}

/* Hands the caller the prompt without the prefix that was just pre-processed. */
static void publish_stripped_prompt(const char *prompt_text, const char *vis_prefix, char *buf,
                                    size_t cap, const char **out_prompt) {
    const char *stripped = prompt_text;
    size_t pfx_len       = strlen(vis_prefix);
    if (strncmp(stripped, vis_prefix, pfx_len) == 0) stripped += pfx_len;
    if (!buf || cap == 0) return;
    snprintf(buf, cap, "%s", stripped);
    if (out_prompt) *out_prompt = buf;
}

static void prefill_chat_prefix(const Config *cfg, const TransformerWeights *w, RunState *s,
                                const MoEConfig *mc, Tokenizer *tok, ThreadPool *tp,
                                const char *prompt_text, char *stripped_prompt_buf,
                                size_t stripped_prompt_buf_cap, const char **out_prompt) {
    if (!wants_visual_prefix(prompt_text, tok)) return;
    int im_start_id        = tokenizer_find_id(tok, "<|im_start|>");
    const char *vis_prefix = (im_start_id >= 0) ? "<|im_start|>User: " : "User: ";
    int prefix_toks[32];
    int n_pre = tokenizer_encode(tok, vis_prefix, strlen(vis_prefix), prefix_toks, 32);
    if (n_pre <= 0) return;
    for (int pi = 0; pi < n_pre; pi++) {
        transformer_forward(prefix_toks[pi], s->current_pos, cfg, w, s, mc, tp);
        s->current_pos++;
    }
    publish_stripped_prompt(prompt_text, vis_prefix, stripped_prompt_buf, stripped_prompt_buf_cap,
                            out_prompt);
}

/* Weights, image, patches, encoder and projector, in order; 0 at the first stage that fails. */
static int run_stages(VisionModel *vm, VisionBuffers *buf, const char *image_path,
                      const char *vision_path, const char *proj_path, const Config *cfg,
                      ThreadPool *tp) {
    if (!load_vision_weights(vm, vision_path, proj_path)) return 0;
    if (!stage_load_image(vm, image_path, buf)) return 0;
    if (!stage_extract_patches(vm, buf)) return 0;
    if (!stage_encode(vm, tp, buf)) return 0;
    return stage_project(vm, cfg, tp, buf);
}

TernaryError vision_pipeline_run(const char *image_path, const char *vision_path,
                                 const char *proj_path, Config *cfg, TransformerWeights *w,
                                 RunState *s, const MoEConfig *mc, Tokenizer *tok, ThreadPool *tp,
                                 const char *prompt_text, char *stripped_prompt_buf,
                                 size_t stripped_prompt_buf_cap, const char **out_prompt) {
    if (out_prompt) *out_prompt = prompt_text; /* default: unchanged */

    if (!vision_path || !proj_path) {
        TN_LOG_ERR("vision",
                   "[vision] --image requires --vision <vision.bin> and --proj <projector.bin>\n"
                   "[vision] Extract weights first:\n"
                   "[vision]   python tools/extract_multimodal.py --repo moondream-hf/moondream2 "
                   "--out models/\n");
        return TN_ERR_INVALID_ARGS;
    }

    printf("\nVision pipeline: loading %s\n", image_path);

    VisionModel vm;
    memset(&vm, 0, sizeof(vm));
    VisionBuffers buf;
    memset(&buf, 0, sizeof(buf));
    TernaryError result = TN_ERR_IMAGE_LOAD;

    if (run_stages(&vm, &buf, image_path, vision_path, proj_path, cfg, tp)) {
        prefill_chat_prefix(cfg, w, s, mc, tok, tp, prompt_text, stripped_prompt_buf,
                            stripped_prompt_buf_cap, out_prompt);
        VisionContext vctx;
        vctx.patch_embeddings = buf.projected;
        vctx.num_patches      = buf.out_tokens;
        vctx.embed_dim        = cfg->dim;
        inject_vision_into_kv_cache(s, cfg, w, &vctx, tp);
        printf("  Vision context injected (%d KV tokens)\n\n", buf.out_tokens);
        result = TN_OK;
    }

    vision_buffers_free(&buf);
    vision_model_free(&vm);
    return result;
}
