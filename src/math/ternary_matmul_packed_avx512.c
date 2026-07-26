#include "core/platform.h"

#if TN_HAS_AVX512

#include "math/ternary_matmul_packed.h"
#include <immintrin.h>
#include <string.h>

/**
 * AVX-512 fused unpack+matmul for 2-bit packed ternary weights.
 *
 * Processes 16 weights at once (vs 8 for AVX2) using 512-bit registers.
 * Reads 4 packed bytes → 16 ternary weights {-1, 0, +1} per iteration.
 *
 * The inner loop converts ternary integers to float via _mm512_cvtepi32_ps
 * then uses FMA, avoiding masked-move overhead and keeping the pipeline full.
 */

static inline __m512i unpack16_to_epi32(const tn_u8 *row, int j)
{
    /*
     * 16 weights occupy 32 bits (4 bytes): 2 bits each at positions 0,2,...,30.
     * We broadcast those 32 bits into all 16 lanes, then variable-shift each
     * lane right by its bit-offset, mask the low 2 bits, and subtract 1
     * to map the stored encoding {0→-1, 1→0, 2→+1}.
     */
    uint64_t raw = 0;
    int start_byte = j >> 2;
    int end_byte = (j + 15) >> 2;
    size_t bytes_to_copy = (size_t)(end_byte - start_byte + 1);
    memcpy(&raw, row + start_byte, bytes_to_copy);

    uint32_t bits = (uint32_t)(raw >> ((j & 3) << 1));

    __m512i v      = _mm512_set1_epi32((int)bits);
    __m512i shifts = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14,
                                        16, 18, 20, 22, 24, 26, 28, 30);
    v = _mm512_srlv_epi32(v, shifts);
    v = _mm512_and_si512(v, _mm512_set1_epi32(3));
    return _mm512_sub_epi32(v, _mm512_set1_epi32(1));
}

/*
 * How far ahead (in rows) to software-prefetch weight rows.
 * At n=2560, each row is 640 bytes and takes ~270 ns to compute.
 * DRAM latency is ~150 ns, so 4 rows ahead gives ~1 µs lookahead —
 * comfortably hiding memory latency.  Larger n rows take longer to
 * compute, so hardware prefetch takes over naturally for hidden_dim.
 */
#define TN_PREFETCH_ROWS 8

void ternary_matmul_packed_avx512(float *out, const float *x, const tn_u8 *packed_w,
                                   int n, int d, const float *scales, int group_size)
{
    size_t row_bytes = ((size_t)n + 3) >> 2;

    for (int i = 0; i < d; i++) {
        const tn_u8 *row = packed_w + (size_t)i * row_bytes;

        /* Prefetch the first cache line of a future row to warm up the
         * hardware prefetcher for that row's sequential stream.  Without
         * this hint the CPU stalls ~37% of the time waiting for DRAM on
         * the cross-row boundary. */
        if (i + TN_PREFETCH_ROWS < d) {
            _mm_prefetch((const char *)(packed_w + (size_t)(i + TN_PREFETCH_ROWS) * row_bytes),
                         _MM_HINT_T1);
        }

        if (group_size <= 0) {
            /* -------- per-matrix scale -------- */
            __m512 acc512 = _mm512_setzero_ps();
            int j = 0;

            /* 16-wide AVX-512 loop */
            for (; j + 15 < n; j += 16) {
                __m512  xv  = _mm512_loadu_ps(&x[j]);
                __m512i w32 = unpack16_to_epi32(row, j);
                __m512  wf  = _mm512_cvtepi32_ps(w32);
                acc512 = _mm512_fmadd_ps(wf, xv, acc512);
            }

            float val = _mm512_reduce_add_ps(acc512);

            /* Scalar tail (up to 15 elements) */
            for (; j < n; j++) {
                int w = (int)((row[j >> 2] >> ((j & 3) << 1)) & 3) - 1;
                val += (float)w * x[j];
            }

            out[i] = val * scales[0];

        } else {
            /* -------- per-group scale -------- */
            int n_groups = (n + group_size - 1) / group_size;
            float total  = 0.0f;

            for (int g = 0; g < n_groups; g++) {
                int g_start = g * group_size;
                int g_end   = g_start + group_size;
                if (g_end > n) g_end = n;

                __m512 acc512 = _mm512_setzero_ps();
                int j = g_start;

                for (; j + 15 < g_end; j += 16) {
                    __m512  xv  = _mm512_loadu_ps(&x[j]);
                    __m512i w32 = unpack16_to_epi32(row, j);
                    __m512  wf  = _mm512_cvtepi32_ps(w32);
                    acc512 = _mm512_fmadd_ps(wf, xv, acc512);
                }

                float partial = _mm512_reduce_add_ps(acc512);

                for (; j < g_end; j++) {
                    int w = (int)((row[j >> 2] >> ((j & 3) << 1)) & 3) - 1;
                    partial += (float)w * x[j];
                }

                total += partial * scales[i * n_groups + g];
            }

            out[i] = total;
        }
    }
}

#endif /* TN_HAS_AVX512 */
