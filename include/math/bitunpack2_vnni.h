#ifndef TN_BITUNPACK2_VNNI_H
#define TN_BITUNPACK2_VNNI_H

#include "core/platform.h"

#if TN_HAS_AVX512VNNI
#include <immintrin.h>
#include "math/cpu_features.h"

/*
 * Unpack 64 packed 2-bit codes (stored in 16 bytes) into 64 uint8 lanes,
 * returned in a __m512i. Shared by every VNNI kernel over this project's
 * 2-bit ternary encoding (0/1/2 -> -1/0/+1, "w_enc = w+1"), which is bit-
 * for-bit identical between this project's native packed-ternary format
 * (ternary_matmul_packed_vnni.c) and GGUF Q2_0 (matmul_q2_0_vnni.c) — both
 * pack 4 codes/byte, 2 bits each, low-to-high (see either caller's header
 * comment for the derivation). Originally written for the former and
 * extracted here so the latter can reuse it verbatim instead of
 * maintaining a second copy of the same bit-trick.
 *
 * Two implementations, chosen at RUNTIME (not just compile time):
 *
 * 1. VBMI path (Ice Lake+, Sapphire Rapids+): 3 instructions using
 *    vpermb (byte permute) + vpmultishiftqb (per-qword multishift) + AND.
 *    ~2.7x faster than the SSE unpack path in microbenchmarks.
 *
 * 2. SSE path (fallback): 128-bit shifts + masks + byte interleave.
 *    Works on any AVX-512 VNNI CPU without VBMI (e.g., Cascade Lake).
 *
 * Selection is gated by `tn_cpu_features_detect()->avx512vbmi`, NOT just
 * the `TN_HAS_AVX512VBMI` compile-time macro: a hypervisor can advertise
 * AVX-512VBMI via CPUID (matching what `-march=native` detects at compile
 * time too) while the underlying execution unit cannot actually retire the
 * instruction — confirmed on a real Firecracker microVM host, where this
 * used to crash with SIGILL on every first-time calibration run (see
 * docs/ai/mistakes.md). `tn_cpu_features_detect()` execution-verifies VBMI
 * under a SIGILL trap before setting this bit, so both variants must be
 * compiled in whenever VBMI *could* be present, with the choice deferred
 * to that verified runtime flag.
 */

#if TN_HAS_AVX512VBMI

/*
 * VBMI fast path: 3-instruction unpack.
 *
 * Step 1: vpermb replicates each of 16 packed bytes 4x -> 64 bytes in __m512i.
 *         Layout: each qword = {p[2k]x4, p[2k+1]x4}.
 * Step 2: vpmultishiftqb extracts 2-bit fields at positions {0,2,4,6,32,34,36,38}
 *         within each 64-bit group. Produces 8 bytes per qword, 64 total.
 * Step 3: AND 0x03 masks to 2 bits (multishift extracts full 8-bit windows).
 *
 * Correctness: for qword with bytes {B0,B0,B0,B0,B1,B1,B1,B1}:
 *   64-bit value (LE) = B0 | (B0<<8) | (B0<<16) | (B0<<24) | (B1<<32) | ...
 *   shift=0  -> bits[7:0]   = B0        -> &0x03 = B0[1:0] = w_enc[4x(2k)]
 *   shift=2  -> bits[9:2]   = B0>>2 ... -> &0x03 = B0[3:2] = w_enc[4x(2k)+1]
 *   shift=32 -> bits[39:32] = B1        -> &0x03 = B1[1:0] = w_enc[4x(2k+1)]
 */
static const __m512i tn_bitunpack2_vbmi_perm_idx = {
    (long long)0x0101010100000000LL, (long long)0x0303030302020202LL,
    (long long)0x0505050504040404LL, (long long)0x0707070706060606LL,
    (long long)0x0909090908080808LL, (long long)0x0b0b0b0b0a0a0a0aLL,
    (long long)0x0d0d0d0d0c0c0c0cLL, (long long)0x0f0f0f0f0e0e0e0eLL
};

static const __m512i tn_bitunpack2_vbmi_shift_ctrl = {
    (long long)0x2624222006040200LL, (long long)0x2624222006040200LL,
    (long long)0x2624222006040200LL, (long long)0x2624222006040200LL,
    (long long)0x2624222006040200LL, (long long)0x2624222006040200LL,
    (long long)0x2624222006040200LL, (long long)0x2624222006040200LL
};

static inline __m512i tn_unpack64_to_wenc_u8_vbmi(const tn_u8 *row_j)
{
    __m128i p = _mm_loadu_si128((const __m128i *)row_j);
    __m512i expanded = _mm512_permutexvar_epi8(tn_bitunpack2_vbmi_perm_idx,
                                                _mm512_castsi128_si512(p));
    __m512i shifted = _mm512_multishift_epi64_epi8(tn_bitunpack2_vbmi_shift_ctrl, expanded);
    return _mm512_and_si512(shifted, _mm512_set1_epi8(0x03));
}

#endif /* TN_HAS_AVX512VBMI */

/*
 * SSE unpack path: 128-bit shifts + masks + byte interleave. Always compiled
 * (needs only baseline AVX-512F/VNNI, not VBMI) — this is the fallback used
 * whenever the verified runtime check below says VBMI isn't safe to use,
 * and the only path compiled at all on a build without VBMI support.
 *
 * Correctness proof for _mm_srli_epi16 + AND 0x03:
 *   For a 16-bit word (H,L): (HL >> k) & 0x03 yields bits [k+1:k] of L for
 *   k in {0,2,4,6}, since H contamination lands in bits 7:6 which 0x03 masks out.
 *
 * Interleaving order:
 *   m0[i] = w_enc[4i], m1[i] = w_enc[4i+1], m2[i] = w_enc[4i+2], m3[i] = w_enc[4i+3]
 *   unpacklo8/unpackhi8 then unpacklo16/unpackhi16 restores sequential order.
 */
static inline __m512i tn_unpack64_to_wenc_u8_sse(const tn_u8 *row_j)
{
    __m128i p = _mm_loadu_si128((const __m128i *)row_j);

    const __m128i mask2 = _mm_set1_epi8(0x03);

    __m128i m0 = _mm_and_si128(p,                      mask2);
    __m128i m1 = _mm_and_si128(_mm_srli_epi16(p, 2),   mask2);
    __m128i m2 = _mm_and_si128(_mm_srli_epi16(p, 4),   mask2);
    __m128i m3 = _mm_and_si128(_mm_srli_epi16(p, 6),   mask2);

    __m128i lo01 = _mm_unpacklo_epi8(m0, m1);
    __m128i hi01 = _mm_unpackhi_epi8(m0, m1);
    __m128i lo23 = _mm_unpacklo_epi8(m2, m3);
    __m128i hi23 = _mm_unpackhi_epi8(m2, m3);

    __m128i q0 = _mm_unpacklo_epi16(lo01, lo23);
    __m128i q1 = _mm_unpackhi_epi16(lo01, lo23);
    __m128i q2 = _mm_unpacklo_epi16(hi01, hi23);
    __m128i q3 = _mm_unpackhi_epi16(hi01, hi23);

    __m512i wenc = _mm512_castsi128_si512(q0);
    wenc = _mm512_inserti32x4(wenc, q1, 1);
    wenc = _mm512_inserti32x4(wenc, q2, 2);
    wenc = _mm512_inserti32x4(wenc, q3, 3);
    return wenc;
}

/* Runtime dispatch: only take the VBMI path if the CPUID bit was also
 * execution-verified (see cpu_features.c) — never just the compile-time
 * capability. Cached after the first call, same pattern as tn_simd_init(). */
static inline __m512i tn_unpack64_to_wenc_u8(const tn_u8 *row_j)
{
#if TN_HAS_AVX512VBMI
    static int s_use_vbmi = -1;
    if (s_use_vbmi < 0) s_use_vbmi = tn_cpu_features_detect()->avx512vbmi ? 1 : 0;
    if (s_use_vbmi) return tn_unpack64_to_wenc_u8_vbmi(row_j);
#endif
    return tn_unpack64_to_wenc_u8_sse(row_j);
}

#endif /* TN_HAS_AVX512VNNI */

#endif /* TN_BITUNPACK2_VNNI_H */
