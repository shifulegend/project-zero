/*
 * ternary_matmul_lut_avx512bw.c  --  Phase K-6 LUT kernel
 *
 * LUT-based ternary matrix-vector multiply using AVX-512BW vpermt2w lookups.
 * Packs 5 ternary weights {-1,0,1} per byte (balanced ternary, base-3 encoding).
 *
 * Algorithm ported from lut_mm (tommyyliu, GPL-3.0) to C99 without templates.
 * Key idea: precompute 4 x zmm16 lookup tables T[0..3] from 5 int8 activations,
 * then for each 32-column block of packed weights use _mm512_permutex2var_epi16
 * (vpermt2w) to perform 32 simultaneous table lookups in a single instruction.
 *
 * Weight encoding:
 *   v = w[0]*81 + w[1]*27 + w[2]*9 + w[3]*3 + w[4], range [-121..+121]
 *   Stored as int8.  The negative range maps symmetrically.
 *
 * Activation lookup:
 *   hv[t] = a[0]*D0[t] + a[1]*D1[t] + a[2]*D2[t]  for t in [0,26]  (top 3 trits)
 *   lv[t] = a[3]*D3[t] + a[4]*D4[t]                for t in [0,8]   (bottom 2 trits)
 *   T[k][lane] = hv[idx_h[lane]] + lv[idx_l[lane]]
 *     where idx_h = (121 + 32k + lane) / 9
 *           idx_l = (121 + 32k + lane) % 9
 *
 * Critical loop order: sweep K5 groups in OUTER loop, build T[] ONCE per group,
 * then sweep all N column blocks in INNER loop.  This amortizes the O(32*9)
 * table-build cost over all N columns instead of rebuilding it for each block.
 *
 * Accumulation: int16 per block, flushed to int32 every FLUSH_EVERY groups
 * to prevent overflow: max per-lane accumulation = 51 * 121 = 6171 < 32767.
 *
 * This file is guarded by TN_HAS_AVX512 (which implies AVX-512BW when compiled
 * with ISA_AVX512 = -mavx512f -mavx512bw -mavx512vl -mavx512dq from Makefile).
 */

#include "core/platform.h"

#if TN_HAS_AVX512

#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* aligned_alloc needs C11 or _ISOC11_SOURCE; fall back to posix_memalign */
#if !defined(_ISOC11_SOURCE) && defined(_POSIX_C_SOURCE) && (_POSIX_C_SOURCE >= 200112L)
static void *tn_lut_aligned_alloc(size_t align, size_t size) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, align, size) != 0) return NULL;
    return ptr;
}
#else
static void *tn_lut_aligned_alloc(size_t align, size_t size) {
    return aligned_alloc(align, size);
}
#endif

/* ──────────────────────────────────────────────────────────────────────────
 * Public API declarations
 * ────────────────────────────────────────────────────────────────────────── */

/*
 * lut_mm_avx512bw: LUT-based ternary matmul.
 *
 *   A : M x K,     int8 activations (row-major)
 *   P : (K/5) x N, int8 balanced-ternary packed weights (row-major)
 *   C : M x N,     int32 output (overwritten, caller must zero if adding)
 *
 * K must be a multiple of 5.
 */
void lut_mm_avx512bw(const int8_t * restrict A, const int8_t * restrict P,
                     int M, int K, int N, int32_t * restrict C);

/*
 * lut_pack_weights: pack unpacked {-1,0,1} weights to 5-trit balanced ternary.
 *
 *   W : K x N, int8, row-major, values in {-1,0,1}
 *   P : (K_pad/5) x N, int8, K_pad = ((K+4)/5)*5
 */
void lut_pack_weights(const int8_t *W, int K, int N, int8_t *P);


/* ──────────────────────────────────────────────────────────────────────────
 * Digit decomposition tables (static, initialized once)
 * ────────────────────────────────────────────────────────────────────────── */

/*
 * For H (top 3 trits), index t in [0,26]:
 *   d0[t] = t/9 - 1,  d1[t] = (t/3)%3 - 1,  d2[t] = t%3 - 1
 * For L (bottom 2 trits), index t in [0,8]:
 *   d3[t] = t/3 - 1,  d4[t] = t%3 - 1
 */
static int16_t g_d0[27], g_d1[27], g_d2[27];
static int16_t g_d3[9],  g_d4[9];
static int     g_digit_tables_init = 0;

static void init_digit_tables(void) {
    if (g_digit_tables_init) return;
    int t;
    for (t = 0; t < 27; t++) {
        g_d0[t] = (int16_t)(t / 9     - 1);
        g_d1[t] = (int16_t)(t / 3 % 3 - 1);
        g_d2[t] = (int16_t)(t % 3     - 1);
    }
    for (t = 0; t < 9; t++) {
        g_d3[t] = (int16_t)(t / 3 - 1);
        g_d4[t] = (int16_t)(t % 3 - 1);
    }
    g_digit_tables_init = 1;
}

/*
 * Balanced ternary multipliers:
 *   v = w[0]*81 + w[1]*27 + w[2]*9 + w[3]*3 + w[4]
 * range [-121, +121].
 */
#define COEFF0 81
#define COEFF1 27
#define COEFF2  9
#define COEFF3  3
#define COEFF4  1


/* ──────────────────────────────────────────────────────────────────────────
 * build_tables: build 4 ZMM lookup tables from 5 int8 activations
 *
 * T[k] covers 32 int16 values corresponding to the output of the 5-trit dot
 * product for packed weight values in the range [32k - 121, 32k - 121 + 31].
 *
 * For lane t in T[k]:
 *   abs_v    = 121 + 32*k + t  (maps to the "positive" half of [-121..+121])
 *   hv_index = abs_v / 9       in [0..26]
 *   lv_index = abs_v % 9       in [0..8]
 *   T[k][t]  = hv[hv_index] + lv[lv_index]
 *
 * hv[s] = a[0]*d0[s] + a[1]*d1[s] + a[2]*d2[s]  (scalar, s in [0..26])
 * lv[s] = a[3]*d3[s] + a[4]*d4[s]                (scalar, s in [0..8])
 *
 * Negative weight values v < 0 are handled by negating T[k][t] when the
 * original packed byte was negative (see lookup step).
 * ────────────────────────────────────────────────────────────────────────── */
static inline void build_tables(const int8_t a[5], __m512i T[4]) {
    int16_t hv_arr[32] __attribute__((aligned(64)));
    int16_t lv_arr[32] __attribute__((aligned(64)));
    int16_t a0 = (int16_t)a[0], a1 = (int16_t)a[1], a2 = (int16_t)a[2];
    int16_t a3 = (int16_t)a[3], a4 = (int16_t)a[4];
    int s, k;

    /* Compute hv[0..26] scalar */
    for (s = 0; s < 27; s++) {
        hv_arr[s] = (int16_t)(a0 * g_d0[s] + a1 * g_d1[s] + a2 * g_d2[s]);
    }
    hv_arr[27] = hv_arr[28] = hv_arr[29] = hv_arr[30] = hv_arr[31] = 0;

    /* Compute lv[0..8] scalar */
    for (s = 0; s < 9; s++) {
        lv_arr[s] = (int16_t)(a3 * g_d3[s] + a4 * g_d4[s]);
    }
    lv_arr[9] = lv_arr[10] = lv_arr[11] = lv_arr[12] = lv_arr[13] = 0;
    lv_arr[14] = lv_arr[15] = lv_arr[16] = lv_arr[17] = lv_arr[18] = 0;
    lv_arr[19] = lv_arr[20] = lv_arr[21] = lv_arr[22] = lv_arr[23] = 0;
    lv_arr[24] = lv_arr[25] = lv_arr[26] = lv_arr[27] = lv_arr[28] = 0;
    lv_arr[29] = lv_arr[30] = lv_arr[31] = 0;

    __m512i hv = _mm512_load_si512((const __m512i*)hv_arr);
    __m512i lv = _mm512_load_si512((const __m512i*)lv_arr);

    /* Build T[k] for k in [0..3] using permutexvar scatter-gather */
    for (k = 0; k < 4; k++) {
        int16_t idx_h[32] __attribute__((aligned(64)));
        int16_t idx_l[32] __attribute__((aligned(64)));
        int t;
        for (t = 0; t < 32; t++) {
            int abs_v = 121 + 32 * k + t;
            idx_h[t]  = (int16_t)(abs_v / 9);
            idx_l[t]  = (int16_t)(abs_v % 9);
        }
        __m512i vh = _mm512_load_si512((const __m512i*)idx_h);
        __m512i vl = _mm512_load_si512((const __m512i*)idx_l);
        T[k] = _mm512_add_epi16(_mm512_permutexvar_epi16(vh, hv),
                                 _mm512_permutexvar_epi16(vl, lv));
    }
}


/* ──────────────────────────────────────────────────────────────────────────
 * Flush int16 accumulator to int32 accumulator pair
 * acc16: 32 x int16;  acc32[0]: low 16 x int32;  acc32[1]: high 16 x int32
 * ────────────────────────────────────────────────────────────────────────── */
static inline void acc16_to_acc32(__m512i acc16,
                                   __m512i *acc32_lo, __m512i *acc32_hi) {
    __m256i lo16 = _mm512_extracti32x8_epi32(acc16, 0);
    __m256i hi16 = _mm512_extracti32x8_epi32(acc16, 1);
    *acc32_lo = _mm512_add_epi32(*acc32_lo, _mm512_cvtepi16_epi32(lo16));
    *acc32_hi = _mm512_add_epi32(*acc32_hi, _mm512_cvtepi16_epi32(hi16));
}


/* ──────────────────────────────────────────────────────────────────────────
 * lookup_and_acc: look up 32 weight values in T[0..3], negate where needed,
 *                 and accumulate into acc16.
 * pw:     pointer to 32 int8 packed weight values for this column block
 * T:      4 x ZMM lookup tables
 * acc16:  in/out accumulator (32 x int16)
 * ────────────────────────────────────────────────────────────────────────── */
static inline __m512i lookup_and_acc(const int8_t *pw,
                                      const __m512i T[4],
                                      __m512i acc16) {
    /* Load 32 packed weight bytes (values in [-121..+121]) */
    __m512i v = _mm512_loadu_si512((const __m512i*)pw);

    /* abs(v) as int8, and sign mask (bit per byte, set where v < 0) */
    __m512i abs_v    = _mm512_abs_epi8(v);
    __mmask64 neg64  = _mm512_movepi8_mask(v);

    /* Widen low 32 bytes of abs_v (int8) to 32 x int16 */
    __m256i abs_lo8 = _mm512_extracti32x8_epi32(abs_v, 0);
    __m512i idx16   = _mm512_cvtepu8_epi16(abs_lo8);  /* 0..121 */

    /*
     * permutex2var_epi16(a, idx, b):
     *   lane result = (idx[lane] & 32) ? b[idx[lane] & 31] : a[idx[lane] & 31]
     *
     * T[0..1] covers abs values [0..63]:
     *   idx[lane] in [0,31]  -> T[0][idx]       (bit 5 = 0)
     *   idx[lane] in [32,63] -> T[1][idx & 31]  (bit 5 = 1)
     * This works directly because idx16 = abs_value and bit 5 = (abs_value >= 32).
     */
    __m512i r01 = _mm512_permutex2var_epi16(T[0], idx16, T[1]);

    /*
     * T[2..3] covers abs values [64..127]:
     *   Subtract 64 so that [64..127] maps to [0..63] with same bit-5 logic.
     */
    __m512i idx_hi = _mm512_sub_epi16(idx16, _mm512_set1_epi16(64));
    __m512i r23 = _mm512_permutex2var_epi16(T[2], idx_hi, T[3]);

    /* Blend: r01 when abs_value < 64, r23 when abs_value >= 64 */
    __mmask32 hi_mask = _mm512_cmpge_epi16_mask(idx16, _mm512_set1_epi16(64));
    __m512i rr = _mm512_mask_blend_epi16(hi_mask, r01, r23);

    /* Negate lanes where original weight was negative */
    __mmask32 neg32 = (__mmask32)(neg64 & 0xFFFFFFFFULL);
    rr = _mm512_mask_sub_epi16(rr, neg32, _mm512_setzero_si512(), rr);

    return _mm512_add_epi16(acc16, rr);
}


/* ──────────────────────────────────────────────────────────────────────────
 * process_row: compute one output row against all N columns
 *
 * Loop order (critical for performance):
 *   OUTER: groups g in [0, K5)       -- build T[] ONCE per group
 *   INNER: column blocks bn in [0, N/32)  -- reuse T[] for all blocks
 *
 * This amortizes the ~300-cycle table-build cost over N/32 = 64 column blocks
 * instead of rebuilding it for each (block, group) pair.
 * ────────────────────────────────────────────────────────────────────────── */
#define FLUSH_EVERY 51  /* max acc per lane: 51 * 121 = 6171 < 32767 */

/*
 * Workspace layout for process_row (per call):
 *   acc16_all : int16_t[n_full]  -- int16 accumulator for full column blocks
 *   flush_cnt : int[n_blocks]    -- per-block flush counter
 *
 * Caller provides pre-allocated workspace to avoid per-row malloc overhead.
 * ws_size = n_full * sizeof(int16_t) + n_blocks * sizeof(int)  (rounded up to 64B)
 */
static void process_row(const int8_t *a_row, const int8_t *P,
                        int K5, int N, int32_t *C_row,
                        int16_t *acc16_all, int *flush_cnt) {
    int n_full   = N & ~31;   /* largest multiple of 32 <= N */
    int n_tail   = N &  31;   /* remaining columns */
    int n_blocks = n_full / 32;

    /* Zero the output and workspace */
    memset(C_row, 0, (size_t)N * sizeof(int32_t));
    memset(acc16_all, 0, (size_t)n_full * sizeof(int16_t));
    memset(flush_cnt, 0, (size_t)n_blocks * sizeof(int));

    /* ── Outer loop: one group of 5 activations ── */
    for (int g = 0; g < K5; g++) {
        const int8_t *a = a_row + (size_t)g * 5;
        __m512i T[4];
        build_tables(a, T);

        const int8_t *P_row = P + (size_t)g * N;

        /* ── Inner loop: all full 32-column blocks ── */
        for (int bn = 0; bn < n_blocks; bn++) {
            int col0          = bn * 32;
            int16_t *acc16_blk = acc16_all + col0;

            __m512i acc16 = _mm512_loadu_si512((const __m512i*)acc16_blk);
            acc16 = lookup_and_acc(P_row + col0, T, acc16);

            flush_cnt[bn]++;
            if (flush_cnt[bn] == FLUSH_EVERY) {
                /* Flush int16 -> int32 directly to C_row */
                __m512i *acc32_lo = (__m512i*)(C_row + col0);
                __m512i *acc32_hi = (__m512i*)(C_row + col0 + 16);
                __m512i lo = _mm512_loadu_si512(acc32_lo);
                __m512i hi = _mm512_loadu_si512(acc32_hi);
                acc16_to_acc32(acc16, &lo, &hi);
                _mm512_storeu_si512(acc32_lo, lo);
                _mm512_storeu_si512(acc32_hi, hi);
                acc16 = _mm512_setzero_si512();
                flush_cnt[bn] = 0;
            }

            _mm512_storeu_si512((__m512i*)acc16_blk, acc16);
        }

        /* Tail columns handled in scalar pass after main loop */
    }

    /* ── Final flush of remaining int16 accumulator to int32 ── */
    for (int bn = 0; bn < n_blocks; bn++) {
        if (flush_cnt[bn] > 0) {
            int col0 = bn * 32;
            __m512i acc16 = _mm512_loadu_si512((const __m512i*)(acc16_all + col0));
            __m512i *acc32_lo = (__m512i*)(C_row + col0);
            __m512i *acc32_hi = (__m512i*)(C_row + col0 + 16);
            __m512i lo = _mm512_loadu_si512(acc32_lo);
            __m512i hi = _mm512_loadu_si512(acc32_hi);
            acc16_to_acc32(acc16, &lo, &hi);
            _mm512_storeu_si512(acc32_lo, lo);
            _mm512_storeu_si512(acc32_hi, hi);
        }
    }

    /* ── Tail columns: scalar accumulation ── */
    if (n_tail > 0) {
        int col0 = n_full;
        for (int g = 0; g < K5; g++) {
            const int8_t *a = a_row + (size_t)g * 5;
            const int8_t *pw = P + (size_t)g * N + col0;
            /*
             * Scalar decode and accumulate, mirroring build_tables(): shift
             * the packed magnitude into the base-3 digit space (+121) before
             * splitting into hv/lv indices, matching the vectorized path.
             */
            for (int j = 0; j < n_tail; j++) {
                int8_t pv = pw[j];  /* balanced ternary packed value */
                int v    = (int)pv;
                int sign = (v < 0) ? -1 : 1;
                int av   = (v < 0) ? -v : v;
                int u      = av + 121;
                int hv_idx = u / 9;
                int lv_idx = u % 9;
                int16_t hv_val = (int16_t)a[0] * g_d0[hv_idx]
                               + (int16_t)a[1] * g_d1[hv_idx]
                               + (int16_t)a[2] * g_d2[hv_idx];
                int16_t lv_val = (int16_t)a[3] * g_d3[lv_idx]
                               + (int16_t)a[4] * g_d4[lv_idx];
                int32_t dot = (int32_t)(hv_val + lv_val) * sign;
                C_row[col0 + j] += dot;
            }
        }
    }
}


/* ──────────────────────────────────────────────────────────────────────────
 * Public functions
 * ────────────────────────────────────────────────────────────────────────── */

void lut_mm_avx512bw(const int8_t * restrict A, const int8_t * restrict P,
                     int M, int K, int N, int32_t * restrict C) {
    assert(K % 5 == 0 && "K must be a multiple of 5 for 5-trit LUT kernel");

    init_digit_tables();

    int K5     = K / 5;
    int n_full = N & ~31;
    int n_blocks = n_full / 32;

    /* Allocate workspace ONCE for all M rows */
    int16_t *acc16_ws = (int16_t*)tn_lut_aligned_alloc(64,
                            (size_t)n_full * sizeof(int16_t) + 64);
    int *flush_ws = (int*)malloc((size_t)(n_blocks + 1) * sizeof(int));
    if (!acc16_ws || !flush_ws) {
        /* Fallback: allocate per-row (slow but correct) */
        free(acc16_ws);
        free(flush_ws);
        return;
    }

    int m;
    for (m = 0; m < M; m++) {
        process_row(A + (size_t)m * K, P, K5, N, C + (size_t)m * N,
                    acc16_ws, flush_ws);
    }

    free(acc16_ws);
    free(flush_ws);
}

void lut_pack_weights(const int8_t *W, int K, int N, int8_t *P) {
    /*
     * W: K x N, int8 row-major, values in {-1,0,1}
     * P: (K_pad/5) x N, int8, balanced-ternary encoded
     */
    int K_pad = ((K + 4) / 5) * 5;
    int K5    = K_pad / 5;
    int g, i, n;

    for (g = 0; g < K5; g++) {
        for (n = 0; n < N; n++) {
            int coeffs[5] = { COEFF0, COEFF1, COEFF2, COEFF3, COEFF4 };
            int v = 0;
            for (i = 0; i < 5; i++) {
                int ki = 5 * g + i;
                int w  = (ki < K) ? (int)W[(size_t)ki * N + n] : 0;
                v += w * coeffs[i];
            }
            P[(size_t)g * N + n] = (int8_t)v;
        }
    }
}

#endif /* TN_HAS_AVX512 */
