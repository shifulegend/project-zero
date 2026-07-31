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
#include "multimodal/image_load.h"
#include "multimodal/patch_extract.h"
#include "multimodal/vision_encoder.h"
#include "multimodal/vision_projector.h"
#include "multimodal/vision_bridge.h"
#include "multimodal/vision_weights_load.h"
#include "memory/aligned_alloc.h"
#include "transformer/forward.h"
#include <stdio.h>
#include <string.h>

TernaryError vision_pipeline_run(const char *image_path,
                                  const char *vision_path,
                                  const char *proj_path,
                                  Config *cfg,
                                  TransformerWeights *w,
                                  RunState *s,
                                  const MoEConfig *mc,
                                  Tokenizer *tok,
                                  ThreadPool *tp,
                                  const char *prompt_text,
                                  char *stripped_prompt_buf,
                                  size_t stripped_prompt_buf_cap,
                                  const char **out_prompt) {
    if (out_prompt) *out_prompt = prompt_text; /* default: unchanged */

    if (!vision_path || !proj_path) {
        fprintf(stderr,
            "[vision] --image requires --vision <vision.bin> and --proj <projector.bin>\n"
            "[vision] Extract weights first:\n"
            "[vision]   python tools/extract_multimodal.py --repo moondream-hf/moondream2 --out models/\n");
        return TN_ERR_INVALID_ARGS;
    }

    printf("\nVision pipeline: loading %s\n", image_path);

    /* Load vision model weights */
    VisionModel vm;
    memset(&vm, 0, sizeof(vm));
    int vm_ok = 0;
    TernaryError result = TN_ERR_IMAGE_LOAD;

    if (vision_model_load_encoder(&vm, vision_path) == TN_OK &&
        vision_model_load_projector(&vm, proj_path) == TN_OK) {
        vision_model_print_info(&vm);
        vm_ok = 1;
    } else {
        fprintf(stderr, "[vision] failed to load vision weights\n");
    }

    if (vm_ok) {
        /* Load and normalize image */
        float *pixels = NULL;
        int img_w = 0, img_h = 0;
        int target_res = (int)(0.5f + __builtin_sqrtf((float)vm.cfg.num_patches))
                         * (int)(0.5f + __builtin_sqrtf((float)(vm.cfg.patch_dim / 3)));
        if (target_res <= 0) target_res = 384;

        if (load_image(image_path, &pixels, &img_w, &img_h, target_res) != TN_OK) {
            fprintf(stderr, "[vision] failed to load image: %s\n", image_path);
            vm_ok = 0;
        } else {
            printf("  Image loaded: %dx%d (SigLIP-normalized)\n", img_w, img_h);
        }

        if (vm_ok) {
            /* Extract patches */
            int patch_size = (int)(0.5f + __builtin_sqrtf((float)(vm.cfg.patch_dim / 3)));
            int max_patches = vm.cfg.num_patches;
            float *patches = (float *)tn_aligned_alloc(
                (size_t)max_patches * vm.cfg.patch_dim * sizeof(float), 64);
            int num_patches = 0;

            if (!patches) {
                fprintf(stderr, "[vision] OOM allocating patches\n");
                vm_ok = 0;
            } else {
                extract_patches(pixels, patches, target_res, patch_size, &num_patches);
                printf("  Extracted %d patches (patch_size=%d)\n", num_patches, patch_size);
                vm.cfg.num_patches = num_patches;
            }

            if (vm_ok) {
                /* Vision encoder forward */
                float *vis_emb = (float *)tn_aligned_alloc(
                    (size_t)num_patches * vm.cfg.embed_dim * sizeof(float), 64);
                if (!vis_emb) {
                    fprintf(stderr, "[vision] OOM: vision embeddings\n");
                } else {
                    printf("  Running vision encoder (%d layers)...\n", vm.cfg.n_layers);
                    vision_encoder_forward(vis_emb, patches, &vm.cfg, &vm.weights, tp);
                    printf("  Encoder done.\n");

                    /* MLP projector (or pixel-shuffle + single linear) */
                    int out_tokens = vision_projector_output_tokens(&vm.proj, num_patches);
                    float *projected = (float *)tn_aligned_alloc(
                        (size_t)out_tokens * cfg->dim * sizeof(float), 64);

                    /* Projector llm_dim must match LLM dim; warn if mismatched */
                    if (vm.proj.llm_dim != cfg->dim) {
                        fprintf(stderr,
                            "[vision] WARNING: projector llm_dim=%d != LLM dim=%d\n"
                            "[vision] Embeddings will be truncated/zero-padded.\n",
                            vm.proj.llm_dim, cfg->dim);
                    }

                    if (!projected) {
                        fprintf(stderr, "[vision] OOM: projected embeddings\n");
                    } else {
                        /* Use projector's own llm_dim for the projection */
                        float *proj_buf = (float *)tn_aligned_alloc(
                            (size_t)out_tokens * vm.proj.llm_dim * sizeof(float), 64);
                        if (proj_buf) {
                            vision_projector_forward_batch(proj_buf, vis_emb,
                                                           num_patches, &vm.proj, tp);
                            /* Copy into projected, matching LLM dim */
                            int copy_dim = vm.proj.llm_dim < cfg->dim ? vm.proj.llm_dim : cfg->dim;
                            memset(projected, 0, (size_t)out_tokens * cfg->dim * sizeof(float));
                            for (int pp = 0; pp < out_tokens; pp++) {
                                memcpy(&projected[pp * cfg->dim],
                                       &proj_buf[pp * vm.proj.llm_dim],
                                       (size_t)copy_dim * sizeof(float));
                            }
                            tn_aligned_free(proj_buf);
                            printf("  Projector done: %d tokens x %d-dim  (scale_factor=%d)\n",
                                   out_tokens, cfg->dim, vm.proj.scale_factor);

                            /* For ChatML-style models (chat_template contains "im_start"),
                             * pre-process a chat prefix so visual tokens land at the correct
                             * sequence position (after "User: "), matching training layout.
                             * Detected from the model's own chat_template — no vocab_size
                             * heuristics. */
                            if (prompt_text && tok->chat_template &&
                                strstr(tok->chat_template, "im_start")) {
                                int im_start_id = tokenizer_find_id(tok, "<|im_start|>");
                                const char *vis_prefix = (im_start_id >= 0)
                                    ? "<|im_start|>User: "
                                    : "User: ";
                                int prefix_toks[32];
                                int n_pre = tokenizer_encode(tok, vis_prefix, strlen(vis_prefix),
                                                             prefix_toks, 32);
                                if (n_pre > 0) {
                                    for (int pi = 0; pi < n_pre; pi++) {
                                        transformer_forward(prefix_toks[pi], s->current_pos,
                                                            cfg, w, s, mc, tp);
                                        s->current_pos++;
                                    }
                                    const char *stripped = prompt_text;
                                    size_t pfx_len = strlen(vis_prefix);
                                    if (strncmp(stripped, vis_prefix, pfx_len) == 0)
                                        stripped += pfx_len;
                                    if (stripped_prompt_buf && stripped_prompt_buf_cap > 0) {
                                        snprintf(stripped_prompt_buf, stripped_prompt_buf_cap, "%s", stripped);
                                        if (out_prompt) *out_prompt = stripped_prompt_buf;
                                    }
                                }
                            }

                            VisionContext vctx;
                            vctx.patch_embeddings = projected;
                            vctx.num_patches      = out_tokens;
                            vctx.embed_dim        = cfg->dim;
                            inject_vision_into_kv_cache(s, cfg, w, &vctx, tp);
                            printf("  Vision context injected (%d KV tokens)\n\n",
                                   out_tokens);
                            result = TN_OK;
                        }
                        tn_aligned_free(projected);
                    }
                    tn_aligned_free(vis_emb);
                }
                tn_aligned_free(patches);
            }
            tn_aligned_free(pixels);
        }
    }

    vision_model_free(&vm);
    return result;
}
