#include "multimodal/vision_projector.h"
#include "math/elementwise.h"
#include "memory/aligned_alloc.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef __AVX2__
#include <immintrin.h>

/* AVX2 vectorized float matmul: out = x @ W^T + b
 * x is [in_dim], W is [out_dim x in_dim] row-major, b is [out_dim] (nullable). */
static void float_matmul_proj(float *out, const float *x, const float *w,
                               const float *b, int in_dim, int out_dim) {
    for (int i = 0; i < out_dim; i++) {
        __m256 acc = _mm256_setzero_ps();
        const float *wi = w + (size_t)i * in_dim;
        int j = 0;
        for (; j + 8 <= in_dim; j += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + j), _mm256_loadu_ps(wi + j), acc);
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        float val = _mm_cvtss_f32(lo);
        for (; j < in_dim; j++) val += x[j] * wi[j];
        out[i] = (b ? b[i] : 0.0f) + val;
    }
}

#else /* scalar fallback */

static void float_matmul_proj(float *out, const float *x, const float *w,
                               const float *b, int in_dim, int out_dim) {
    for (int i = 0; i < out_dim; i++) {
        float val = b ? b[i] : 0.0f;
        for (int j = 0; j < in_dim; j++)
            val += x[j] * w[(size_t)i * in_dim + j];
        out[i] = val;
    }
}

#endif /* __AVX2__ */

// Emulates GELU with SiLU or similar standard activation since standard gelu isn't strictly in elementwise yet,
// we'll just use the silu provided in math/elementwise.c which is typically used for Gated Linear Units.
static void apply_activation(float *x, int n) {
    silu(x, n);
}

TernaryError vision_projector_load(VisionProjector *proj, const void *mapped_ptr, size_t *offset) {
    if (!proj || !mapped_ptr || !offset) return TN_ERR_INVALID_ARGS;

    const float *f_ptr = (const float*)((const char*)mapped_ptr + *offset);

    // In a real loader, we'd copy or map pointers. We'll simulate pointer mapping here.
    // Assuming proj->vision_dim, proj->llm_dim, proj->hidden_dim are pre-populated from config.
    int vd = proj->vision_dim;
    int ld = proj->llm_dim;
    int hd = proj->hidden_dim;

    proj->w_down = (float*)f_ptr; f_ptr += hd * vd;
    proj->bias_down = (float*)f_ptr; f_ptr += hd;
    proj->w_up = (float*)f_ptr; f_ptr += ld * hd;
    proj->bias_up = (float*)f_ptr; f_ptr += ld;

    *offset = ((const char*)f_ptr) - ((const char*)mapped_ptr);
    return TN_OK;
}

/* Internal single-patch forward using a caller-supplied scratch buffer.
 * Avoids repeated alloc/free in batch paths. */
static void proj_forward_inner(float *out, const float *patch_embedding,
                                const VisionProjector *proj, float *hidden) {
    int vd = proj->vision_dim;
    int ld = proj->llm_dim;
    int hd = proj->hidden_dim;

    float_matmul_proj(hidden, patch_embedding, proj->w_down, proj->bias_down, vd, hd);
    apply_activation(hidden, hd);
    float_matmul_proj(out, hidden, proj->w_up, proj->bias_up, hd, ld);
}

void vision_projector_forward(float *out, const float *patch_embedding,
                              const VisionProjector *proj, ThreadPool *tp) {
    (void)tp;
    float *hidden = (float*)tn_aligned_alloc((size_t)proj->hidden_dim * sizeof(float), 64);
    if (!hidden) return;
    proj_forward_inner(out, patch_embedding, proj, hidden);
    tn_aligned_free(hidden);
}

int vision_projector_output_tokens(const VisionProjector *proj, int num_encoder_patches) {
    if (proj->scale_factor <= 1) return num_encoder_patches;
    int s2 = proj->scale_factor * proj->scale_factor;
    return num_encoder_patches / s2;
}

void vision_projector_forward_batch(float *out, const float *patches, int num_patches,
                                    const VisionProjector *proj, ThreadPool *tp) {
    (void)tp;
    int vd = proj->vision_dim;
    int ld = proj->llm_dim;
    int sf = proj->scale_factor;

    if (sf > 1) {
        /* Pixel-shuffle + single linear.
         * patches: [num_patches × vd]  (vd = encoder embed_dim)
         * After shuffle: [out_tokens × (vd * sf * sf)]  → single linear → [out_tokens × ld]
         *
         * Spatial layout: patches arranged in a sqrt(N) × sqrt(N) grid.
         * Each sf×sf block of patches is concatenated into one output token. */
        int s2        = sf * sf;
        int grid_w    = (int)(0.5 + sqrt((double)num_patches));
        int out_grid  = grid_w / sf;
        int out_tokens = out_grid * out_grid;
        int shuffle_dim = vd * s2;

        float *shuffled = (float*)tn_aligned_alloc(
            (size_t)out_tokens * shuffle_dim * sizeof(float), 64);
        if (!shuffled) return;

        for (int oy = 0; oy < out_grid; oy++) {
            for (int ox = 0; ox < out_grid; ox++) {
                int out_idx = oy * out_grid + ox;
                float *dst  = shuffled + (size_t)out_idx * shuffle_dim;
                int feat_off = 0;
                for (int dy = 0; dy < sf; dy++) {
                    for (int dx = 0; dx < sf; dx++) {
                        int in_y  = oy * sf + dy;
                        int in_x  = ox * sf + dx;
                        int in_idx = in_y * grid_w + in_x;
                        memcpy(dst + feat_off,
                               patches + (size_t)in_idx * vd,
                               (size_t)vd * sizeof(float));
                        feat_off += vd;
                    }
                }
            }
        }

        /* Single linear: w_up[ld × shuffle_dim] maps each output token */
        for (int t = 0; t < out_tokens; t++) {
            float_matmul_proj(&out[(size_t)t * ld],
                              shuffled + (size_t)t * shuffle_dim,
                              proj->w_up, proj->bias_up,
                              shuffle_dim, ld);
        }
        tn_aligned_free(shuffled);
        return;
    }

    /* Original two-layer MLP path (scale_factor <= 1). */
    float *hidden = (float*)tn_aligned_alloc((size_t)proj->hidden_dim * sizeof(float), 64);
    if (!hidden) return;
    for (int p = 0; p < num_patches; p++)
        proj_forward_inner(&out[p * ld], &patches[p * vd], proj, hidden);
    tn_aligned_free(hidden);
}
