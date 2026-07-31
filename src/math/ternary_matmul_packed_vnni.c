#include "core/platform.h"

#if TN_HAS_AVX512VNNI

#include "math/ternary_matmul_packed.h"
#include "math/quantize_i8.h"
#include "math/bitunpack2_vnni.h"
#include <immintrin.h>
#include <string.h>
#include <stdint.h>

/*
 * AVX-512 VNNI ternary matmul (Phase 16-S.2 — K-1a)
 *
 * Replaces the float32 FMA kernel with int8 VNNI for a ~2× speedup in the
 * compute-bound phase and ~1.5× practical speedup when DRAM-limited.
 *
 * Algorithm ("w_enc bias trick"):
 *   Packed 2-bit weight encoding: 0 → ternary -1, 1 → ternary 0, 2 → ternary +1
 *   Therefore: raw 2-bit value  = w_enc = w + 1  ∈ {0, 1, 2}  (uint8-safe)
 *
 *   dpbusds(w_enc_u8, q_x_i8) = Σ(w_enc[j] * q_x[j])
 *                               = Σ((w[j]+1) * q_x[j])
 *                               = dot(w, q_x) + Σ q_x[j]
 *
 *   true_dot = dpbusds_result - sum_qx          (bias correction)
 *   out[i]   = true_dot * act_scale * w_scale[i]  (dequantization)
 *
 * sum_qx is computed once per matmul call (same q_x vector for all d rows).
 *
 * Saturation analysis:
 *   w_enc ∈ {0,1,2}, q_x ∈ [-127,127] → per-group-of-4 max = 4*2*127 = 1016
 *   n=5632 → max acc = 5632/4 * 1016 = 1,430,528 << INT32_MAX = no saturation.
 *
 * Memory:
 *   Stack-allocates up to TN_VNNI_MAX_N int8s for q_x.
 *   Adjust if n ever exceeds 16384.
 */

#define TN_VNNI_MAX_N 16384
#define TN_PREFETCH_ROWS  8   /* primary lookahead (≈1 DRAM latency) */

/*
 * Prefetch ALL cache lines of the target weight row.
 * A packed row for n=2560 is 640 bytes = 10 cache lines.
 * Issuing only one _mm_prefetch() per row leaves 9/10 cache lines un-prefetched,
 * causing DRAM stalls when the inner loop reaches those lines.
 * This macro issues a prefetch for every 64-byte cache line in the row.
 */
#define TN_PREFETCH_ROW_ALL(ptr, row_bytes, hint)                          \
    do {                                                                    \
        const char *_p = (const char *)(ptr);                              \
        size_t _rb = (row_bytes);                                           \
        for (size_t _off = 0; _off < _rb; _off += 64)                     \
            _mm_prefetch(_p + _off, (hint));                               \
    } while (0)

/*
 * Unpack 64 packed 2-bit weights (stored in 16 bytes) into 64 uint8 w_enc
 * values {0, 1, 2}, returned in a __m512i. VBMI-fast-path-or-SSE-fallback
 * implementation now lives in math/bitunpack2_vnni.h (tn_unpack64_to_wenc_u8),
 * shared with matmul_q2_0_vnni.c's identical bit layout — see that header
 * for the derivation.
 */

void ternary_matmul_packed_vnni(float *out, const float *x, const tn_u8 *packed_w,
                                 int n, int d, const float *scales, int group_size)
{
    if (n > TN_VNNI_MAX_N) {
        /* Safety fallback — should never happen with current models */
        extern void ternary_matmul_packed_avx512(float*, const float*, const tn_u8*,
                                                  int, int, const float*, int);
        ternary_matmul_packed_avx512(out, x, packed_w, n, d, scales, group_size);
        return;
    }

    /* ── Step 1: Quantize activations (once per matmul call) ── */
    int8_t  q_x[TN_VNNI_MAX_N];
    float   act_scale = quantize_row_to_i8_avx512(x, q_x, n);

    if (act_scale == 0.0f) {
        /* All-zero input — output is zero regardless of weights */
        memset(out, 0, (size_t)d * sizeof(float));
        return;
    }

    /* ── Step 2: sum_qx for bias correction (once per matmul call) ── */
    int32_t sum_qx = sum_i8_avx512(q_x, n);

    /* ── Step 3: Per-row VNNI dot product ── */
    size_t row_bytes = ((size_t)n + 3) >> 2;

    if (group_size <= 0) {
        /* ── Per-matrix scale ── */
        float w_scale = scales[0];

        for (int i = 0; i < d; i++) {
            const tn_u8 *row = packed_w + (size_t)i * row_bytes;

            if (i + TN_PREFETCH_ROWS < d)
                TN_PREFETCH_ROW_ALL(packed_w + (size_t)(i + TN_PREFETCH_ROWS) * row_bytes,
                                    row_bytes, _MM_HINT_T1);


            __m512i acc = _mm512_setzero_si512();
            int j = 0;

            /* 64-weight VNNI loop */
            for (; j + 63 < n; j += 64) {
                /* Unpack 64 weights from 16 packed bytes → 64 uint8 w_enc */
                __m512i wenc = tn_unpack64_to_wenc_u8(row + (j >> 2));

                /* Load 64 int8 activations */
                __m512i qxv = _mm512_loadu_si512((const __m512i*)(q_x + j));

                /* VNNI: acc += Σ(wenc[k] * qxv[k]) in groups of 4 → 16 int32 */
                acc = _mm512_dpbusds_epi32(acc, wenc, qxv);
            }

            /* Reduce 16-lane int32 accumulator to scalar */
            int32_t vnni_result = _mm512_reduce_add_epi32(acc);

            /* Scalar tail (< 64 remaining weights) */
            for (; j < n; j++) {
                int w_enc = (int)((row[j >> 2] >> ((j & 3) << 1)) & 3);
                vnni_result += w_enc * (int32_t)q_x[j];
            }

            /* Bias correction + dequantize */
            int32_t true_dot = vnni_result - sum_qx;
            out[i] = (float)true_dot * act_scale * w_scale;
        }

    } else {
        /* ── Per-group scale ── */
        int n_groups = (n + group_size - 1) / group_size;

        /* Precompute per-group partial sums of q_x for bias correction */
        int32_t sum_qx_group[1024];  /* supports up to 1024 groups */
        int safe_groups = n_groups < 1024 ? n_groups : 1024;
        for (int g = 0; g < safe_groups; g++) {
            int gs = g * group_size;
            int ge = gs + group_size;
            if (ge > n) ge = n;
            sum_qx_group[g] = sum_i8(q_x + gs, ge - gs);
        }

        for (int i = 0; i < d; i++) {
            const tn_u8 *row = packed_w + (size_t)i * row_bytes;

            if (i + TN_PREFETCH_ROWS < d)
                TN_PREFETCH_ROW_ALL(packed_w + (size_t)(i + TN_PREFETCH_ROWS) * row_bytes,
                                    row_bytes, _MM_HINT_T1);


            float total = 0.0f;

            for (int g = 0; g < safe_groups; g++) {
                int gs = g * group_size;
                int ge = gs + group_size;
                if (ge > n) ge = n;

                __m512i acc = _mm512_setzero_si512();
                int j = gs;

                for (; j + 63 < ge; j += 64) {
                    __m512i wenc = tn_unpack64_to_wenc_u8(row + (j >> 2));

                    __m512i qxv = _mm512_loadu_si512((const __m512i*)(q_x + j));
                    acc = _mm512_dpbusds_epi32(acc, wenc, qxv);
                }

                int32_t vnni_result = _mm512_reduce_add_epi32(acc);

                for (; j < ge; j++) {
                    int w_enc = (int)((row[j >> 2] >> ((j & 3) << 1)) & 3);
                    vnni_result += w_enc * (int32_t)q_x[j];
                }

                int32_t true_dot = vnni_result - sum_qx_group[g];
                total += (float)true_dot * act_scale * scales[i * n_groups + g];
            }

            out[i] = total;
        }
    }
}

/*
 * K-4 R-3: Pre-quantised VNNI entry point.
 *
 * Same algorithm as ternary_matmul_packed_vnni() but skips Steps 1 and 2
 * (activation quantisation and sum_qx calculation). The caller provides
 * already-quantised activations (q_x), the dequantisation scale (act_scale),
 * and the precomputed bias-correction sum (sum_qx).
 *
 * Used by parallel_ternary_matmul_packed() which quantises x ONCE before
 * dispatching all worker slices, eliminating:
 *   - N_threads redundant quantisations per matmul dispatch (e.g., 4× at T=4)
 *   - Additional redundancy across Q/K/V projections that share the same x
 *
 * Prefetch: uses _MM_HINT_T0 (L1 hint) at 4 rows ahead — tighter lookahead
 * than the standard path (T1/8 rows) since smaller lookahead fits better in
 * L1 (48 KB Tiger Lake) alongside the 2.5 KB activation vector.
 */
void ternary_matmul_packed_vnni_preq(float *out,
                                      const int8_t *q_x, float act_scale,
                                      int32_t sum_qx,
                                      const tn_u8 *packed_w,
                                      int n, int d,
                                      const float *scales, int group_size)
{
    if (act_scale == 0.0f) {
        memset(out, 0, (size_t)d * sizeof(float));
        return;
    }

    size_t row_bytes = ((size_t)n + 3) >> 2;

    if (group_size <= 0) {
        float w_scale = scales[0];

        for (int i = 0; i < d; i++) {
            const tn_u8 *row = packed_w + (size_t)i * row_bytes;

            if (i + TN_PREFETCH_ROWS < d)
                TN_PREFETCH_ROW_ALL(packed_w + (size_t)(i + TN_PREFETCH_ROWS) * row_bytes,
                                    row_bytes, _MM_HINT_T1);


            __m512i acc = _mm512_setzero_si512();
            int j = 0;

            for (; j + 63 < n; j += 64) {
                __m512i wenc = tn_unpack64_to_wenc_u8(row + (j >> 2));

                __m512i qxv = _mm512_loadu_si512((const __m512i*)(q_x + j));
                acc = _mm512_dpbusds_epi32(acc, wenc, qxv);
            }

            int32_t vnni_result = _mm512_reduce_add_epi32(acc);

            for (; j < n; j++) {
                int w_enc = (int)((row[j >> 2] >> ((j & 3) << 1)) & 3);
                vnni_result += w_enc * (int32_t)q_x[j];
            }

            int32_t true_dot = vnni_result - sum_qx;
            out[i] = (float)true_dot * act_scale * w_scale;
        }

    } else {
        int n_groups = (n + group_size - 1) / group_size;

        int32_t sum_qx_group[1024];
        int safe_groups = n_groups < 1024 ? n_groups : 1024;
        for (int g = 0; g < safe_groups; g++) {
            int gs = g * group_size;
            int ge = gs + group_size;
            if (ge > n) ge = n;
            sum_qx_group[g] = sum_i8(q_x + gs, ge - gs);
        }

        for (int i = 0; i < d; i++) {
            const tn_u8 *row = packed_w + (size_t)i * row_bytes;

            if (i + TN_PREFETCH_ROWS < d)
                TN_PREFETCH_ROW_ALL(packed_w + (size_t)(i + TN_PREFETCH_ROWS) * row_bytes,
                                    row_bytes, _MM_HINT_T1);

            float total = 0.0f;

            for (int g = 0; g < safe_groups; g++) {
                int gs = g * group_size;
                int ge = gs + group_size;
                if (ge > n) ge = n;

                __m512i acc = _mm512_setzero_si512();
                int j = gs;

                for (; j + 63 < ge; j += 64) {
                    __m512i wenc = tn_unpack64_to_wenc_u8(row + (j >> 2));

                    __m512i qxv = _mm512_loadu_si512((const __m512i*)(q_x + j));
                    acc = _mm512_dpbusds_epi32(acc, wenc, qxv);
                }

                int32_t vnni_result = _mm512_reduce_add_epi32(acc);

                for (; j < ge; j++) {
                    int w_enc = (int)((row[j >> 2] >> ((j & 3) << 1)) & 3);
                    vnni_result += w_enc * (int32_t)q_x[j];
                }

                int32_t true_dot = vnni_result - sum_qx_group[g];
                total += (float)true_dot * act_scale * scales[i * n_groups + g];
            }

            out[i] = total;
        }
    }
}

#endif /* TN_HAS_AVX512VNNI */
