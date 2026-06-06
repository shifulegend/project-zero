#include "multimodal/vision_weights_load.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* ── Internal: read a uint32 at byte offset in mapped data ─────────────── */
static uint32_t rd32(const void *base, size_t off) {
    uint32_t v;
    memcpy(&v, (const char *)base + off, 4);
    return v;
}

static int32_t rd32s(const void *base, size_t off) {
    int32_t v;
    memcpy(&v, (const char *)base + off, 4);
    return v;
}

/* ── Convenience: pointer to float array inside mmap ─────────────────────
 * Returns a float* at byte offset |off| into |base|.
 * Caller is responsible for not exceeding mapped bounds. */
static float *fptr(const void *base, size_t off) {
    return (float *)((const char *)base + off);
}

/* ── Vision encoder loader ──────────────────────────────────────────────── */
TernaryError vision_model_load_encoder(VisionModel *vm, const char *path) {
    if (!vm || !path) return TN_ERR_INVALID_ARGS;

    TernaryError err = mapped_file_open(&vm->vision_mf, path);
    if (err != TN_OK) {
        fprintf(stderr, "[vision] cannot open %s\n", path);
        return err;
    }

    const void *d  = vm->vision_mf.data;
    size_t      sz = vm->vision_mf.size;

    if (sz < 64) { fprintf(stderr, "[vision] file too small\n"); goto fail; }

    uint32_t magic = rd32(d, 0);
    if (magic != VISION_BIN_MAGIC) {
        fprintf(stderr, "[vision] bad magic 0x%08X (expected 0x%08X)\n",
                magic, VISION_BIN_MAGIC);
        goto fail;
    }

    /* uint32_t version = rd32(d, 4); — reserved for future */
    int n_layers    = rd32s(d, 8);
    int embed_dim   = rd32s(d, 12);
    int hidden_dim  = rd32s(d, 16);
    int n_heads     = rd32s(d, 20);
    int patch_dim   = rd32s(d, 24);
    int num_patches = rd32s(d, 28);

    if (n_layers < 1 || n_layers > 128 ||
        embed_dim < 1 || embed_dim > 8192 ||
        n_heads   < 1 || n_heads   > 64   ||
        patch_dim < 1 || num_patches < 1) {
        fprintf(stderr, "[vision] invalid config in header\n");
        goto fail;
    }

    vm->cfg.n_layers    = n_layers;
    vm->cfg.embed_dim   = embed_dim;
    vm->cfg.hidden_dim  = hidden_dim;
    vm->cfg.n_heads     = n_heads;
    vm->cfg.patch_dim   = patch_dim;
    vm->cfg.num_patches = num_patches;

    /* Weight data starts at byte 64 (header padded to 64-byte boundary) */
    size_t off = 64;

    VisionWeights *w = &vm->weights;
    memset(w, 0, sizeof(*w));

    /* Flat arrays — point directly into mmap */
#define ADVANCE(field, count) \
    do { \
        size_t nbytes = (size_t)(count) * sizeof(float); \
        if (off + nbytes > sz) { fprintf(stderr, "[vision] truncated at " #field "\n"); goto fail; } \
        w->field = fptr(d, off); \
        off += nbytes; \
    } while (0)

    ADVANCE(patch_proj_w,    (size_t)embed_dim * patch_dim);
    ADVANCE(patch_proj_b,    (size_t)embed_dim);
    ADVANCE(pos_embed,       (size_t)num_patches * embed_dim);
    ADVANCE(rms_final_weight,(size_t)embed_dim);

#undef ADVANCE

    /* Per-layer weight arrays */
    w->rms_att_weight = (float **)calloc((size_t)n_layers, sizeof(float *));
    w->rms_ffn_weight = (float **)calloc((size_t)n_layers, sizeof(float *));
    w->wq = (float **)calloc((size_t)n_layers, sizeof(float *));
    w->wk = (float **)calloc((size_t)n_layers, sizeof(float *));
    w->wv = (float **)calloc((size_t)n_layers, sizeof(float *));
    w->wo = (float **)calloc((size_t)n_layers, sizeof(float *));
    w->w1 = (float **)calloc((size_t)n_layers, sizeof(float *));
    w->w2 = (float **)calloc((size_t)n_layers, sizeof(float *));
    w->w3 = (float **)calloc((size_t)n_layers, sizeof(float *));

    if (!w->rms_att_weight || !w->wq || !w->w1) {
        fprintf(stderr, "[vision] OOM allocating per-layer pointer arrays\n");
        goto fail_free;
    }

#define LAYER_PTR(field, count) \
    do { \
        size_t nbytes = (size_t)(count) * sizeof(float); \
        if (off + nbytes > sz) { fprintf(stderr, "[vision] truncated at layer %d " #field "\n", l); goto fail_free; } \
        w->field[l] = fptr(d, off); \
        off += nbytes; \
    } while (0)

    for (int l = 0; l < n_layers; l++) {
        LAYER_PTR(rms_att_weight, embed_dim);
        LAYER_PTR(rms_ffn_weight, embed_dim);
        LAYER_PTR(wq, (size_t)embed_dim * embed_dim);
        LAYER_PTR(wk, (size_t)embed_dim * embed_dim);
        LAYER_PTR(wv, (size_t)embed_dim * embed_dim);
        LAYER_PTR(wo, (size_t)embed_dim * embed_dim);
        LAYER_PTR(w1, (size_t)hidden_dim * embed_dim);
        LAYER_PTR(w2, (size_t)embed_dim  * hidden_dim);
        LAYER_PTR(w3, (size_t)hidden_dim * embed_dim);
    }

#undef LAYER_PTR

    vm->vision_loaded = 1;
    return TN_OK;

fail_free:
    free(w->rms_att_weight); free(w->rms_ffn_weight);
    free(w->wq); free(w->wk); free(w->wv); free(w->wo);
    free(w->w1); free(w->w2); free(w->w3);
fail:
    mapped_file_close(&vm->vision_mf);
    return TN_ERR_INVALID_ARGS;
}

/* ── Projector loader ───────────────────────────────────────────────────── */
TernaryError vision_model_load_projector(VisionModel *vm, const char *path) {
    if (!vm || !path) return TN_ERR_INVALID_ARGS;

    TernaryError err = mapped_file_open(&vm->proj_mf, path);
    if (err != TN_OK) {
        fprintf(stderr, "[vision] cannot open projector %s\n", path);
        return err;
    }

    const void *d  = vm->proj_mf.data;
    size_t      sz = vm->proj_mf.size;

    if (sz < 64) { fprintf(stderr, "[vision] projector file too small\n"); goto fail; }

    uint32_t magic = rd32(d, 0);
    if (magic != PROJECTOR_BIN_MAGIC) {
        fprintf(stderr, "[vision] projector bad magic 0x%08X\n", magic);
        goto fail;
    }

    int vision_dim   = rd32s(d, 8);
    int llm_dim      = rd32s(d, 12);
    int hidden_dim   = rd32s(d, 16);
    int has_bias     = rd32s(d, 20);
    int scale_factor = rd32s(d, 24);  /* 0 or 1 → no pixel shuffle; >1 → pixel shuffle */
    if (scale_factor <= 0) scale_factor = 1;

    if (vision_dim < 1 || llm_dim < 1 || hidden_dim < 0) {
        fprintf(stderr, "[vision] invalid projector config\n");
        goto fail;
    }

    VisionProjector *proj = &vm->proj;
    proj->vision_dim  = vision_dim;
    proj->llm_dim     = llm_dim;
    proj->hidden_dim  = hidden_dim;
    proj->scale_factor = scale_factor;

    size_t off = 64;

#define PROJ_PTR(field, count) \
    do { \
        size_t nbytes = (size_t)(count) * sizeof(float); \
        if (off + nbytes > sz) { fprintf(stderr, "[vision] projector truncated at " #field "\n"); goto fail; } \
        proj->field = fptr(d, off); \
        off += nbytes; \
    } while (0)

    if (hidden_dim == 0) {
        /* Single-linear + pixel-shuffle mode: w_up holds the full linear weight.
         * Input to the linear is vision_dim * scale_factor^2 (post pixel-shuffle). */
        int s2 = scale_factor * scale_factor;
        int shuffle_dim = vision_dim * s2;
        proj->w_down    = NULL;
        proj->bias_down = NULL;
        PROJ_PTR(w_up, (size_t)llm_dim * shuffle_dim);
        if (has_bias) { PROJ_PTR(bias_up, llm_dim); } else { proj->bias_up = NULL; }
    } else {
        /* Two-layer MLP mode. */
        PROJ_PTR(w_down,    (size_t)hidden_dim * vision_dim);
        if (has_bias) { PROJ_PTR(bias_down, hidden_dim); } else { proj->bias_down = NULL; }
        PROJ_PTR(w_up,      (size_t)llm_dim * hidden_dim);
        if (has_bias) { PROJ_PTR(bias_up,   llm_dim); }    else { proj->bias_up   = NULL; }
    }

#undef PROJ_PTR

    vm->proj_loaded = 1;
    return TN_OK;

fail:
    mapped_file_close(&vm->proj_mf);
    return TN_ERR_INVALID_ARGS;
}

void vision_model_print_info(const VisionModel *vm) {
    printf("Vision Model:\n");
    if (vm->vision_loaded) {
        printf("  Encoder: %d layers, embed=%d, hidden=%d, heads=%d, patches=%d\n",
               vm->cfg.n_layers, vm->cfg.embed_dim, vm->cfg.hidden_dim,
               vm->cfg.n_heads, vm->cfg.num_patches);
        printf("  Patch dim: %d  (patch_size=%d px)\n",
               vm->cfg.patch_dim,
               (int)(0.5f + __builtin_sqrtf((float)(vm->cfg.patch_dim / 3))));
    } else {
        printf("  Encoder: not loaded\n");
    }
    if (vm->proj_loaded) {
        printf("  Projector: %d → %d (hidden %d)\n",
               vm->proj.vision_dim, vm->proj.llm_dim, vm->proj.hidden_dim);
    } else {
        printf("  Projector: not loaded\n");
    }
}

void vision_model_free(VisionModel *vm) {
    if (!vm) return;
    if (vm->vision_loaded) {
        free(vm->weights.rms_att_weight);
        free(vm->weights.rms_ffn_weight);
        free(vm->weights.wq); free(vm->weights.wk);
        free(vm->weights.wv); free(vm->weights.wo);
        free(vm->weights.w1); free(vm->weights.w2); free(vm->weights.w3);
        mapped_file_close(&vm->vision_mf);
        vm->vision_loaded = 0;
    }
    if (vm->proj_loaded) {
        mapped_file_close(&vm->proj_mf);
        vm->proj_loaded = 0;
    }
}
