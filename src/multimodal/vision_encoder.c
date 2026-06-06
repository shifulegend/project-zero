#include "multimodal/vision_encoder.h"
#include "math/rmsnorm.h"
#include "math/softmax.h"
#include "math/elementwise.h"
#include "memory/aligned_alloc.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __AVX2__
#include <immintrin.h>

/* AVX2 vectorized float matmul: out = x * W^T + b
 * x is [in_dim], W is [out_dim x in_dim] (row-major), b is [out_dim] (nullable).
 * ~8× faster than scalar for large in_dim (e.g. 768, 3072). */
static void float_matmul(float *out, const float *x, const float *w,
                         const float *b, int in_dim, int out_dim) {
    for (int i = 0; i < out_dim; i++) {
        __m256 acc = _mm256_setzero_ps();
        const float *wi = w + (size_t)i * in_dim;
        int j = 0;
        for (; j + 8 <= in_dim; j += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + j), _mm256_loadu_ps(wi + j), acc);
        /* horizontal reduce */
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        float val = _mm_cvtss_f32(lo);
        /* scalar tail */
        for (; j < in_dim; j++) val += x[j] * wi[j];
        out[i] = (b ? b[i] : 0.0f) + val;
    }
}

#else /* scalar fallback */

/* out = x * W^T + b.   x is [1 x in_dim], W is [out_dim x in_dim], b is [out_dim] */
static void float_matmul(float *out, const float *x, const float *w,
                         const float *b, int in_dim, int out_dim) {
    for (int i = 0; i < out_dim; i++) {
        float val = b ? b[i] : 0.0f;
        for (int j = 0; j < in_dim; j++)
            val += x[j] * w[(size_t)i * in_dim + j];
        out[i] = val;
    }
}

#endif /* __AVX2__ */

// Bidirectional Multi-Head Attention for Vision Encoder
static void vision_attention(float *x, const VisionConfig *cfg, const VisionWeights *w, int layer, float *scratch) {
    int P = cfg->num_patches;
    int D = cfg->embed_dim;
    int H = cfg->n_heads;
    int head_dim = D / H;

    float *q = scratch;
    float *k = q + P * D;
    float *v = k + P * D;
    float *att = v + P * D; // P * P per head, we can process one head at a time
    float *out = att + P * P;

    // Compute Q, K, V for all patches
    for (int p = 0; p < P; p++) {
        float_matmul(&q[p * D], &x[p * D], w->wq[layer], NULL, D, D);
        float_matmul(&k[p * D], &x[p * D], w->wk[layer], NULL, D, D);
        float_matmul(&v[p * D], &x[p * D], w->wv[layer], NULL, D, D);
    }

    // Multi-head attention
    float scale = 1.0f / sqrtf((float)head_dim);
    for (int h = 0; h < H; h++) {
        for (int p = 0; p < P; p++) {
            // Compute scores for patch p to all other patches
            for (int p2 = 0; p2 < P; p2++) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += q[p * D + h * head_dim + d] * k[p2 * D + h * head_dim + d];
                }
                att[p * P + p2] = score * scale;
            }
            softmax(&att[p * P], P); // Standard softmax along last dim

            // Weighted sum of V
            for (int d = 0; d < head_dim; d++) {
                float val = 0.0f;
                for (int p2 = 0; p2 < P; p2++) {
                    val += att[p * P + p2] * v[p2 * D + h * head_dim + d];
                }
                out[p * D + h * head_dim + d] = val;
            }
        }
    }

    for (int p = 0; p < P; p++) {
        float_matmul(&q[p * D], &out[p * D], w->wo[layer], NULL, D, D);
        // The residual is added by the caller, so we just return the attention output in q.
        // Copy q back to out
        for (int d = 0; d < D; d++) out[p * D + d] = q[p * D + d];
    }
}

static void vision_ffn(float *x, const VisionConfig *cfg, const VisionWeights *w, int layer, float *scratch) {
    int P = cfg->num_patches;
    int D = cfg->embed_dim;
    int HD = cfg->hidden_dim;

    float *gate = scratch;
    float *up = gate + P * HD;
    float *out = up + P * HD;

    for (int p = 0; p < P; p++) {
        float_matmul(&gate[p * HD], &x[p * D], w->w1[layer], NULL, D, HD);
        float_matmul(&up[p * HD], &x[p * D], w->w3[layer], NULL, D, HD);

        silu(&gate[p * HD], HD);

        for (int d = 0; d < HD; d++) {
            gate[p * HD + d] *= up[p * HD + d];
        }

        float_matmul(&out[p * D], &gate[p * HD], w->w2[layer], NULL, HD, D);
    }
}

void vision_encoder_forward(float *out_embeddings, const float *patches, const VisionConfig *cfg, const VisionWeights *w, ThreadPool *tp) {
    (void)tp; // Single threaded for simple implementation
    int P = cfg->num_patches;
    int D = cfg->embed_dim;
    int patch_dim = cfg->patch_dim;

    // Compute patch embeddings
    for (int p = 0; p < P; p++) {
        float_matmul(&out_embeddings[p * D], &patches[p * patch_dim], w->patch_proj_w, w->patch_proj_b, patch_dim, D);
        // Add positional embeddings
        for (int d = 0; d < D; d++) {
            out_embeddings[p * D + d] += w->pos_embed[p * D + d];
        }
    }

    size_t scratch_size = 0;
    tn_size_mul3(P, cfg->hidden_dim, 4, &scratch_size);
    if (scratch_size < (size_t)(P * D * 3 + P * P + P * D)) {
        scratch_size = P * D * 3 + P * P + P * D;
    }
    float *scratch = (float*)tn_aligned_alloc(scratch_size * sizeof(float), 64);
    float *normed = (float*)tn_aligned_alloc(P * D * sizeof(float), 64);

    if (!scratch || !normed) {
        if (scratch) tn_aligned_free(scratch);
        if (normed) tn_aligned_free(normed);
        return;
    }

    for (int l = 0; l < cfg->n_layers; l++) {
        // Attention Norm
        for (int p = 0; p < P; p++) rmsnorm(&normed[p * D], &out_embeddings[p * D], w->rms_att_weight[l], D, 1e-6f);

        // Attention (modifies normed to hold attention output or returns it in a scratch buffer)
        // Let's pass 'scratch + P*D*3' or similar. Wait, vision_attention overwrites q,k,v.
        // Actually, vision_attention leaves the final wo output in `out` which is passed via `scratch`.
        // The last loop in vision_attention copies q to out. But out is local to vision_attention.
        // Let's just adjust vision_attention and vision_ffn to output to a specific buffer.

        float *att_out = scratch + P * D * 3 + P * P; // same as 'out' in vision_attention
        vision_attention(normed, cfg, w, l, scratch);
        // Add residual
        for (int p = 0; p < P; p++) {
           for(int d=0; d<D; d++) out_embeddings[p*D+d] += att_out[p*D+d];
        }

        for (int p = 0; p < P; p++) rmsnorm(&normed[p * D], &out_embeddings[p * D], w->rms_ffn_weight[l], D, 1e-6f);

        // FFN
        float *ffn_out = scratch + P * cfg->hidden_dim * 2; // same as 'out' in vision_ffn
        vision_ffn(normed, cfg, w, l, scratch);

        // Add residual
        for (int p = 0; p < P; p++) {
           for(int d=0; d<D; d++) out_embeddings[p*D+d] += ffn_out[p*D+d];
        }
    }

    // Final norm
    for (int p = 0; p < P; p++) rmsnorm(&out_embeddings[p * D], &out_embeddings[p * D], w->rms_final_weight, D, 1e-6f);

    tn_aligned_free(scratch);
    tn_aligned_free(normed);
}
