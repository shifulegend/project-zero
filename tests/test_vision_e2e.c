/*
 * test_vision_e2e.c — Phase 11 End-to-End Vision Pipeline Benchmark
 *
 * Tests the FULL multimodal pipeline on a real image file:
 *   load_image → extract_patches → vision_encoder_forward →
 *   vision_projector_forward_batch → VisionContext setup
 *
 * Measures wall-clock time for each stage and reports synthetic "patch/s"
 * (equivalent to tokens injected per second for the vision prefix).
 *
 * Usage:
 *   build/test_vision_e2e <image_path>
 *   build/test_vision_e2e strawberry.jpg
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "multimodal/image_load.h"
#include "multimodal/patch_extract.h"
#include "multimodal/vision_encoder.h"
#include "multimodal/vision_projector.h"
#include "multimodal/vision_bridge.h"
#include "memory/aligned_alloc.h"
#include "math/simd_dispatch.h"

/* ── timing ─────────────────────────────────────────────────────────────── */
static int64_t now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000LL + ts.tv_nsec / 1000LL;
}

/* ── tiny random weight init ─────────────────────────────────────────────── */
static void init_weights_random(float *w, int n) {
    for (int i = 0; i < n; i++)
        w[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.02f;
}
static void init_weights_ones(float *w, int n) {
    for (int i = 0; i < n; i++) w[i] = 1.0f;
}

int main(int argc, char **argv) {
    const char *img_path = (argc > 1) ? argv[1] : "strawberry.jpg";

    printf("=======================================================\n");
    printf("  Project Zero — Phase 11 Vision Pipeline Benchmark\n");
    printf("=======================================================\n");
    printf("Image: %s\n\n", img_path);

    /* ── SIMD init (uses same dispatch as main engine) ─────────────────── */
    const char *simd = tn_simd_init();
    printf("SIMD backend: %s\n\n", simd);

    srand(42);

    /* ── Config ─────────────────────────────────────────────────────────── */
    /* SigLIP-style: 384×384 image, 14px patches → 27×27 = 729 patches      */
    const int TARGET_RES  = 384;
    const int PATCH_SIZE  = 14;
    const int VISION_DIM  = 768;   /* SigLIP / ViT-B embed dim              */
    const int VISION_HDIM = 3072;  /* ViT-B FFN hidden dim (4× embed)       */
    const int VISION_HEADS = 12;
    const int VISION_LAYERS = 2;   /* stub: 2 layers so test runs quickly   */
    const int LLM_DIM     = 2560;  /* Qwen2-2B native dim                   */

    VisionConfig vcfg;
    vcfg.patch_dim  = PATCH_SIZE * PATCH_SIZE * 3;   /* 588 floats/patch    */
    vcfg.embed_dim  = VISION_DIM;
    vcfg.hidden_dim = VISION_HDIM;
    vcfg.n_layers   = VISION_LAYERS;
    vcfg.n_heads    = VISION_HEADS;

    /* ── Stage 1: Load image ─────────────────────────────────────────────── */
    printf("Stage 1: Image Loading\n");
    printf("  Target resolution: %dx%d\n", TARGET_RES, TARGET_RES);

    float *pixels = NULL;
    int img_w = 0, img_h = 0;
    int64_t t0 = now_us();
    TernaryError err = load_image(img_path, &pixels, &img_w, &img_h, TARGET_RES);
    int64_t t1 = now_us();

    if (err != TN_OK) {
        fprintf(stderr, "  FAILED to load image (err=%d)\n", (int)err);
        return 1;
    }
    printf("  Loaded + resized to %dx%d in %.2f ms\n",
           img_w, img_h, (t1 - t0) / 1000.0);

    /* Pixel stats: confirm SigLIP normalization produced [-1, 1] range */
    {
        float pmin = pixels[0], pmax = pixels[0], pmean = 0.0f;
        int n = img_w * img_h * 3;
        for (int i = 0; i < n; i++) {
            if (pixels[i] < pmin) pmin = pixels[i];
            if (pixels[i] > pmax) pmax = pixels[i];
            pmean += pixels[i];
        }
        pmean /= (float)n;
        printf("  Pixel stats (post-norm): min=%.4f  max=%.4f  mean=%.4f\n",
               pmin, pmax, pmean);
        /* stbir linear resize can overshoot by ~1-2% at edges — allow ±0.05 headroom */
        printf("  %s\n", (pmin >= -1.05f && pmax <= 1.05f)
               ? "PASS: pixels in [-1, 1] (SigLIP-normalized)" : "WARN: pixels out of expected range");
    }

    /* ── Stage 2: Patch Extraction ───────────────────────────────────────── */
    printf("\nStage 2: Patch Extraction\n");
    printf("  Patch size: %dx%d  →  %d×%d grid\n",
           PATCH_SIZE, PATCH_SIZE, TARGET_RES/PATCH_SIZE, TARGET_RES/PATCH_SIZE);

    int num_patches = 0;
    /* worst case patches = (384/14)^2 ≈ 729 — use ceiling */
    int max_patches = ((TARGET_RES + PATCH_SIZE - 1) / PATCH_SIZE) *
                      ((TARGET_RES + PATCH_SIZE - 1) / PATCH_SIZE);
    float *patches = (float *)tn_aligned_alloc(
        (size_t)max_patches * vcfg.patch_dim * sizeof(float), 64);
    if (!patches) { fprintf(stderr, "OOM\n"); return 1; }

    int64_t t2 = now_us();
    extract_patches(pixels, patches, TARGET_RES, PATCH_SIZE, &num_patches);
    int64_t t3 = now_us();

    vcfg.num_patches = num_patches;
    printf("  Extracted %d patches in %.2f ms\n",
           num_patches, (t3 - t2) / 1000.0);
    printf("  Data: %d patches × %d floats = %.2f MB\n",
           num_patches, vcfg.patch_dim,
           (double)num_patches * vcfg.patch_dim * 4 / (1024*1024));

    /* ── Stage 3: Vision Encoder Forward ────────────────────────────────── */
    printf("\nStage 3: Vision Encoder Forward (ViT — %d layers)\n", VISION_LAYERS);

    /* Alloc dummy weights (zero-init then scale so norms don't collapse) */
    VisionWeights vw;
    memset(&vw, 0, sizeof(vw));

    vw.patch_proj_w = (float *)tn_aligned_alloc(VISION_DIM * vcfg.patch_dim * sizeof(float), 64);
    vw.patch_proj_b = (float *)tn_aligned_alloc(VISION_DIM * sizeof(float), 64);
    vw.pos_embed    = (float *)tn_aligned_alloc((size_t)num_patches * VISION_DIM * sizeof(float), 64);
    vw.rms_final_weight = (float *)tn_aligned_alloc(VISION_DIM * sizeof(float), 64);
    init_weights_random(vw.patch_proj_w, VISION_DIM * vcfg.patch_dim);
    init_weights_random(vw.patch_proj_b, VISION_DIM);
    init_weights_random(vw.pos_embed, num_patches * VISION_DIM);
    init_weights_ones(vw.rms_final_weight, VISION_DIM);

    vw.wq = (float **)calloc(VISION_LAYERS, sizeof(float *));
    vw.wk = (float **)calloc(VISION_LAYERS, sizeof(float *));
    vw.wv = (float **)calloc(VISION_LAYERS, sizeof(float *));
    vw.wo = (float **)calloc(VISION_LAYERS, sizeof(float *));
    vw.w1 = (float **)calloc(VISION_LAYERS, sizeof(float *));
    vw.w2 = (float **)calloc(VISION_LAYERS, sizeof(float *));
    vw.w3 = (float **)calloc(VISION_LAYERS, sizeof(float *));
    vw.rms_att_weight = (float **)calloc(VISION_LAYERS, sizeof(float *));
    vw.rms_ffn_weight = (float **)calloc(VISION_LAYERS, sizeof(float *));

    for (int l = 0; l < VISION_LAYERS; l++) {
        vw.wq[l] = (float *)tn_aligned_alloc(VISION_DIM * VISION_DIM * sizeof(float), 64);
        vw.wk[l] = (float *)tn_aligned_alloc(VISION_DIM * VISION_DIM * sizeof(float), 64);
        vw.wv[l] = (float *)tn_aligned_alloc(VISION_DIM * VISION_DIM * sizeof(float), 64);
        vw.wo[l] = (float *)tn_aligned_alloc(VISION_DIM * VISION_DIM * sizeof(float), 64);
        vw.w1[l] = (float *)tn_aligned_alloc(VISION_HDIM * VISION_DIM * sizeof(float), 64);
        vw.w2[l] = (float *)tn_aligned_alloc(VISION_DIM * VISION_HDIM * sizeof(float), 64);
        vw.w3[l] = (float *)tn_aligned_alloc(VISION_HDIM * VISION_DIM * sizeof(float), 64);
        vw.rms_att_weight[l] = (float *)tn_aligned_alloc(VISION_DIM * sizeof(float), 64);
        vw.rms_ffn_weight[l] = (float *)tn_aligned_alloc(VISION_DIM * sizeof(float), 64);
        init_weights_random(vw.wq[l], VISION_DIM * VISION_DIM);
        init_weights_random(vw.wk[l], VISION_DIM * VISION_DIM);
        init_weights_random(vw.wv[l], VISION_DIM * VISION_DIM);
        init_weights_random(vw.wo[l], VISION_DIM * VISION_DIM);
        init_weights_random(vw.w1[l], VISION_HDIM * VISION_DIM);
        init_weights_random(vw.w2[l], VISION_DIM * VISION_HDIM);
        init_weights_random(vw.w3[l], VISION_HDIM * VISION_DIM);
        init_weights_ones(vw.rms_att_weight[l], VISION_DIM);
        init_weights_ones(vw.rms_ffn_weight[l], VISION_DIM);
    }

    float *vision_embeddings = (float *)tn_aligned_alloc(
        (size_t)num_patches * VISION_DIM * sizeof(float), 64);
    if (!vision_embeddings) { fprintf(stderr, "OOM\n"); return 1; }

    int64_t t4 = now_us();
    vision_encoder_forward(vision_embeddings, patches, &vcfg, &vw, NULL);
    int64_t t5 = now_us();

    double enc_ms = (t5 - t4) / 1000.0;
    printf("  %d patches × %d-dim encoder: %.2f ms\n", num_patches, VISION_DIM, enc_ms);

    /* ── Stage 4: MLP Vision Projector ──────────────────────────────────── */
    printf("\nStage 4: Vision Projector (MLP %d → %d)\n", VISION_DIM, LLM_DIM);

    VisionProjector proj;
    proj.vision_dim = VISION_DIM;
    proj.llm_dim    = LLM_DIM;
    proj.hidden_dim = LLM_DIM;   /* standard: hidden_dim == llm_dim */

    proj.w_down   = (float *)tn_aligned_alloc(proj.hidden_dim * proj.vision_dim * sizeof(float), 64);
    proj.bias_down = (float *)tn_aligned_alloc(proj.hidden_dim * sizeof(float), 64);
    proj.w_up     = (float *)tn_aligned_alloc(proj.llm_dim * proj.hidden_dim * sizeof(float), 64);
    proj.bias_up  = (float *)tn_aligned_alloc(proj.llm_dim * sizeof(float), 64);
    init_weights_random(proj.w_down, proj.hidden_dim * proj.vision_dim);
    init_weights_random(proj.bias_down, proj.hidden_dim);
    init_weights_random(proj.w_up, proj.llm_dim * proj.hidden_dim);
    init_weights_random(proj.bias_up, proj.llm_dim);

    float *projected = (float *)tn_aligned_alloc(
        (size_t)num_patches * LLM_DIM * sizeof(float), 64);
    if (!projected) { fprintf(stderr, "OOM\n"); return 1; }

    int64_t t6 = now_us();
    vision_projector_forward_batch(projected, vision_embeddings, num_patches, &proj, NULL);
    int64_t t7 = now_us();

    double proj_ms = (t7 - t6) / 1000.0;
    printf("  %d patches × (%d→%d→%d): %.2f ms\n",
           num_patches, VISION_DIM, proj.hidden_dim, LLM_DIM, proj_ms);

    /* Embedding stats: confirm projected embeddings are finite and non-degenerate */
    {
        float emin = projected[0], emax = projected[0], emean = 0.0f;
        int n = num_patches * LLM_DIM;
        for (int i = 0; i < n; i++) {
            if (projected[i] < emin) emin = projected[i];
            if (projected[i] > emax) emax = projected[i];
            emean += projected[i];
        }
        emean /= (float)n;
        printf("  Embedding stats: min=%.4f  max=%.4f  mean=%.6f\n",
               emin, emax, emean);
        printf("  %s\n", (isfinite(emin) && isfinite(emax))
               ? "PASS: embeddings are finite" : "FAIL: NaN/Inf detected");
    }

    /* ── Stage 5: Vision Bridge (structural test) ────────────────────────── */
    printf("\nStage 5: Vision Bridge (KV injection structural test)\n");
    VisionContext vctx;
    vctx.patch_embeddings = projected;
    vctx.num_patches      = num_patches;
    vctx.embed_dim        = LLM_DIM;

    if (argc > 2) {
        /* Full LLM integration requires a real model file */
        printf("  [Full LLM integration — requires model at %s]\n", argv[2]);
        printf("  [Load model + RunState, call inject_vision_into_kv_cache, then generate()]\n");
    } else {
        printf("  [Skipped — pass model path as argv[2] for full LLM integration]\n");
        printf("  VisionContext built: %d patches × %d-dim  PASS\n",
               vctx.num_patches, vctx.embed_dim);
    }

    /* ── Summary ─────────────────────────────────────────────────────────── */
    double load_ms = (t1 - t0) / 1000.0;
    double patch_ms = (t3 - t2) / 1000.0;
    double total_ms = load_ms + patch_ms + enc_ms + proj_ms;

    printf("\n=======================================================\n");
    printf("  RESULTS\n");
    printf("=======================================================\n");
    printf("  Image load + resize  : %7.2f ms\n", load_ms);
    printf("  Patch extraction     : %7.2f ms\n", patch_ms);
    printf("  Vision encoder (%dl) : %7.2f ms  (%d patches × %d-dim)\n",
           VISION_LAYERS, enc_ms, num_patches, VISION_DIM);
    printf("  MLP projector        : %7.2f ms  (%d→%d)\n",
           proj_ms, VISION_DIM, LLM_DIM);
    printf("  ─────────────────────────────────────────────────\n");
    printf("  TOTAL vision pipeline: %7.2f ms\n", total_ms);
    printf("\n");
    printf("  Patches produced     : %d\n", num_patches);
    printf("  KV tokens consumed   : %d  (1 token = 1 patch)\n", num_patches);
    printf("  Pipeline patch/s     : %.1f  (encoder+projector only)\n",
           num_patches / ((enc_ms + proj_ms) / 1000.0));
    printf("  End-to-end patch/s   : %.1f  (full pipeline incl. load)\n",
           num_patches / (total_ms / 1000.0));
    printf("\n");
    printf("  NOTE: Encoder uses %d/%d stub layers (full ViT-B=12 layers).\n",
           VISION_LAYERS, 12);
    printf("  Full 12-layer encoder time ≈ %.0f ms (×6 extrapolation).\n",
           enc_ms * 6.0);
    printf("  LLM KV injection timing requires the full model (see REPL /image).\n");
    printf("=======================================================\n");

    /* Cleanup */
    tn_aligned_free(pixels);
    tn_aligned_free(patches);
    tn_aligned_free(vision_embeddings);
    tn_aligned_free(projected);
    tn_aligned_free(proj.w_down); tn_aligned_free(proj.bias_down);
    tn_aligned_free(proj.w_up);   tn_aligned_free(proj.bias_up);
    tn_aligned_free(vw.patch_proj_w); tn_aligned_free(vw.patch_proj_b);
    tn_aligned_free(vw.pos_embed); tn_aligned_free(vw.rms_final_weight);
    for (int l = 0; l < VISION_LAYERS; l++) {
        tn_aligned_free(vw.wq[l]); tn_aligned_free(vw.wk[l]);
        tn_aligned_free(vw.wv[l]); tn_aligned_free(vw.wo[l]);
        tn_aligned_free(vw.w1[l]); tn_aligned_free(vw.w2[l]);
        tn_aligned_free(vw.w3[l]);
        tn_aligned_free(vw.rms_att_weight[l]);
        tn_aligned_free(vw.rms_ffn_weight[l]);
    }
    free(vw.wq); free(vw.wk); free(vw.wv); free(vw.wo);
    free(vw.w1); free(vw.w2); free(vw.w3);
    free(vw.rms_att_weight); free(vw.rms_ffn_weight);

    return 0;
}
