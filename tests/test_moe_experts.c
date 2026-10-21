/*
 * tests/test_moe_experts.c — 2026-10-21.
 *
 *  1. q8_0_act integer row dots (Q5_0 / Q5_1 / Q8_0 x Q8_0 activations):
 *     AVX2 path == VNNI path == scalar integer reference (exact integers,
 *     float accumulation within 1e-5 relative).
 *  2. Q4_K row dot: VNNI kernel == AVX2 kernel (same integer partial sums).
 *  3. moe_experts_forward (fused two-dispatch MoE) vs the legacy per-expert
 *     path it replaced in moe_ffn.c (Q4_K gate/up batch, F32-activation
 *     Q5_0 down batch, weighted saxpy, + a Q4_K shared expert): relative L2
 *     error bounded by the Q8_0 activation rounding (< 1e-2), and the fused
 *     result is bit-identical across thread counts (rows are independent).
 */
#include "test_harness.h"
#include "transformer/moe_experts.h"
#include "math/matmul_q4k.h"
#include "math/matmul_q5_0.h"
#include "math/q8_0_act.h"
#include "math/simd_dispatch.h"
#include "core/gguf_reader.h"
#include "threading/thread_pool.h"
#include "sampling/rng.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint16_t f32_to_fp16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t sign = (u >> 16) & 0x8000, exp = (u >> 23) & 0xFF, mant = u & 0x7FFFFF;
    if (exp >= 143) return (uint16_t)(sign | 0x7BFF);
    if (exp <= 112) return (uint16_t)sign;   /* no fp16 subnormals needed here */
    return (uint16_t)(sign | ((exp - 112) << 10) | (mant >> 13));
}
static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1F, mant = h & 0x3FF, bits;
    if (exp == 0) bits = sign;   /* generators below never emit subnormals */
    else bits = sign | ((exp + 112) << 23) | (mant << 13);
    float f; memcpy(&f, &bits, 4); return f;
}

/* random block of `bytes` with an fp16 scale (and optional fp16 min) */
static void fill_legacy(uint8_t *blk, int type, unsigned long long *r) {
    int bytes = type == GGUF_TYPE_Q5_0 ? 22 : type == GGUF_TYPE_Q5_1 ? 24 : 34;
    for (int i = 0; i < bytes; i++) blk[i] = (uint8_t)(rng_float(r) * 256.0f);
    uint16_t d = f32_to_fp16(0.002f + rng_float(r) * 0.02f);
    memcpy(blk, &d, 2);
    if (type == GGUF_TYPE_Q5_1) { uint16_t m = f32_to_fp16(-0.05f + rng_float(r) * 0.1f); memcpy(blk + 2, &m, 2); }
    if (type == GGUF_TYPE_Q8_0) for (int i = 2; i < 34; i++) if ((int8_t)blk[i] == -128) blk[i] = 0x81;
}

/* scalar reference: sum_b d_w * d_a * sum_i q_w[i] * q_a[i] (+ m_w * s_a) in double */
static double ref_dot(const uint8_t *row, int type, const TnQ8_0ActBlock *a, int nb) {
    double sum = 0.0;
    for (int b = 0; b < nb; b++) {
        int bytes = type == GGUF_TYPE_Q5_0 ? 22 : type == GGUF_TYPE_Q5_1 ? 24 : 34;
        const uint8_t *blk = row + (size_t)b * bytes;
        uint16_t dh; memcpy(&dh, blk, 2);
        double dw = fp16_to_f32(dh), mw = 0.0;
        long isum = 0;
        if (type == GGUF_TYPE_Q8_0) {
            for (int i = 0; i < 32; i++) isum += (long)(int8_t)blk[2 + i] * a[b].qs[i];
        } else {
            int off = type == GGUF_TYPE_Q5_1 ? 4 : 2;
            if (type == GGUF_TYPE_Q5_1) { uint16_t mh; memcpy(&mh, blk + 2, 2); mw = fp16_to_f32(mh); }
            uint32_t qh; memcpy(&qh, blk + off, 4);
            const uint8_t *qs = blk + off + 4;
            for (int j = 0; j < 32; j++) {
                int nib = j < 16 ? (qs[j] & 0x0F) : (qs[j - 16] >> 4);
                int q = nib | (int)(((qh >> j) & 1u) << 4);
                if (type == GGUF_TYPE_Q5_0) q -= 16;
                isum += (long)q * a[b].qs[j];
            }
        }
        sum += dw * a[b].d * (double)isum + mw * a[b].s;
    }
    return sum;
}

static void test_q8_0_act_dots(void) {
    unsigned long long r; rng_seed(&r, 4242);
    const int types[3] = { GGUF_TYPE_Q5_0, GGUF_TYPE_Q5_1, GGUF_TYPE_Q8_0 };
    for (int ti = 0; ti < 3; ti++) {
        int type = types[ti];
        int bytes = type == GGUF_TYPE_Q5_0 ? 22 : type == GGUF_TYPE_Q5_1 ? 24 : 34;
        for (int nb = 1; nb <= 44; nb += 7) {
            uint8_t *row = (uint8_t *)malloc((size_t)nb * bytes);
            for (int b = 0; b < nb; b++) fill_legacy(row + (size_t)b * bytes, type, &r);
            float *x = (float *)malloc((size_t)nb * 32 * sizeof(float));
            for (int i = 0; i < nb * 32; i++) x[i] = rng_float(&r) * 4.0f - 2.0f;
            TnQ8_0ActBlock *a = (TnQ8_0ActBlock *)malloc((size_t)nb * sizeof(TnQ8_0ActBlock));
            tn_quantize_q8_0_act(a, x, nb);
            /* activation blocks reproduce x within half a quantization step */
            int qok = 1;
            for (int b = 0; b < nb; b++)
                for (int i = 0; i < 32; i++)
                    if (fabsf(a[b].qs[i] * a[b].d - x[b * 32 + i]) > 0.5001f * a[b].d + 1e-7f) qok = 0;
            TEST_ASSERT(qok, "Q8_0 activation quantization within half a step");
            double ref = ref_dot(row, type, a, nb);
            for (int v = 0; v < 2; v++) {
                float got = type == GGUF_TYPE_Q5_0 ? tn_dot_q5_0_q8_0(row, a, nb, v)
                          : type == GGUF_TYPE_Q5_1 ? tn_dot_q5_1_q8_0(row, a, nb, v)
                          :                          tn_dot_q8_0_q8_0(row, a, nb, v);
                TEST_ASSERT(fabs(got - ref) <= 1e-5 * (1.0 + fabs(ref)),
                            v ? "VNNI row dot matches integer reference"
                              : "AVX2/scalar row dot matches integer reference");
            }
            free(row); free(x); free(a);
        }
    }
}

/* random Q4_K row block with sane fp16 d / dmin */
static void fill_q4k(uint8_t *blk, unsigned long long *r) {
    for (int i = 0; i < 144; i++) blk[i] = (uint8_t)(rng_float(r) * 256.0f);
    uint16_t d = f32_to_fp16(0.0005f + rng_float(r) * 0.002f), m = f32_to_fp16(rng_float(r) * 0.002f);
    memcpy(blk, &d, 2); memcpy(blk + 2, &m, 2);
}

static void test_q4k_vnni_matches_avx2(void) {
    unsigned long long r; rng_seed(&r, 99);
    for (int nb = 1; nb <= 11; nb += 2) {
        uint8_t *row = (uint8_t *)malloc((size_t)nb * 144);
        for (int b = 0; b < nb; b++) fill_q4k(row + (size_t)b * 144, &r);
        float *x = (float *)malloc((size_t)nb * 256 * sizeof(float));
        for (int i = 0; i < nb * 256; i++) x[i] = rng_float(&r) * 2.0f - 1.0f;
        TnQ8KActBlock *a = (TnQ8KActBlock *)malloc((size_t)nb * sizeof(TnQ8KActBlock));
        tn_quantize_q8k(a, x, nb);
        float d0 = tn_dot_q4k_row_q8k(row, a, nb, 0);
        float d1 = tn_dot_q4k_row_q8k(row, a, nb, 1);
        TEST_ASSERT(fabsf(d0 - d1) <= 1e-5f * (1.0f + fabsf(d0)), "Q4_K VNNI dot == AVX2 dot");
        free(row); free(x); free(a);
    }
}

/* ── fused executor vs legacy path ─────────────────────────────────────── */
#define DIM   512
#define EH    96      /* routed expert hidden: multiple of 32, not of 256 */
#define SH    256     /* shared expert hidden: multiple of 256 (Q4_K down) */
#define NEXP  8
#define TOPK  3

static void test_fused_matches_legacy(void) {
    unsigned long long r; rng_seed(&r, 2026);
    const int nb_dim = DIM / 256;
    const size_t q4row = (size_t)nb_dim * 144, q5row = (size_t)(EH / 32) * 22;
    uint8_t *g = (uint8_t *)malloc(q4row * EH * NEXP), *u = (uint8_t *)malloc(q4row * EH * NEXP);
    uint8_t *dn = (uint8_t *)malloc(q5row * DIM * NEXP);
    for (int i = 0; i < EH * NEXP * nb_dim; i++) { fill_q4k(g + (size_t)i * 144, &r); fill_q4k(u + (size_t)i * 144, &r); }
    for (int i = 0; i < DIM * NEXP * (EH / 32); i++) fill_legacy(dn + (size_t)i * 22, GGUF_TYPE_Q5_0, &r);
    /* shared expert: Q4_K gate/up (SH x DIM) and Q4_K down (DIM x SH) */
    uint8_t *sg = (uint8_t *)malloc(q4row * SH), *su = (uint8_t *)malloc(q4row * SH);
    uint8_t *sd = (uint8_t *)malloc((size_t)(SH / 256) * 144 * DIM);
    for (int i = 0; i < SH * nb_dim; i++) { fill_q4k(sg + (size_t)i * 144, &r); fill_q4k(su + (size_t)i * 144, &r); }
    for (int i = 0; i < DIM * (SH / 256); i++) fill_q4k(sd + (size_t)i * 144, &r);

    float x[DIM];
    for (int i = 0; i < DIM; i++) x[i] = rng_float(&r) * 2.0f - 1.0f;
    const int sel[TOPK] = { 5, 1, 6 };
    const float sc[TOPK] = { 0.5f, 0.3f, 0.2f };

    /* legacy: exactly what moe_ffn.c did before 2026-10-21 */
    float ref[DIM]; memset(ref, 0, sizeof(ref));
    {
        TnQ8KActBlock acts[DIM / 256];
        tn_quantize_q8k(acts, x, nb_dim);
        float gb[TOPK][EH], ub[TOPK][EH], db[TOPK][DIM];
        float *gp[TOPK], *up[TOPK], *dp[TOPK];
        const uint8_t *w1[TOPK], *w3[TOPK], *w2[TOPK];
        for (int k = 0; k < TOPK; k++) {
            gp[k] = gb[k]; up[k] = ub[k]; dp[k] = db[k];
            w1[k] = g + (size_t)sel[k] * q4row * EH;
            w3[k] = u + (size_t)sel[k] * q4row * EH;
            w2[k] = dn + (size_t)sel[k] * q5row * DIM;
        }
        parallel_matmul_q4k_batch_preq(gp, acts, w1, DIM, EH, TOPK, NULL);
        parallel_matmul_q4k_batch_preq(up, acts, w3, DIM, EH, TOPK, NULL);
        for (int k = 0; k < TOPK; k++) { tn_silu(gp[k], EH); tn_vec_mul(gp[k], gp[k], up[k], EH); }
        parallel_matmul_q5_0_batch(dp, gp, w2, EH, DIM, TOPK, NULL);
        for (int k = 0; k < TOPK; k++) tn_vec_saxpy(ref, sc[k], dp[k], DIM);
        float hs[SH], hs2[SH], so[DIM];
        parallel_matmul_q4k_preq(hs, acts, sg, DIM, SH, NULL);
        parallel_matmul_q4k_preq(hs2, acts, su, DIM, SH, NULL);
        tn_silu(hs, SH); tn_vec_mul(hs, hs, hs2, SH);
        parallel_matmul_q4k(so, hs, sd, SH, DIM, NULL);
        tn_vec_saxpy(ref, 1.0f, so, DIM);
    }

    TnMoEExpertJob jobs[TOPK + 1];
    for (int k = 0; k < TOPK; k++) {
        TnMoEExpertJob j = { g + (size_t)sel[k] * q4row * EH, u + (size_t)sel[k] * q4row * EH,
                             dn + (size_t)sel[k] * q5row * DIM,
                             GGUF_TYPE_Q4_K, GGUF_TYPE_Q4_K, GGUF_TYPE_Q5_0, EH, sc[k] };
        TEST_ASSERT(moe_experts_job_ok(&j, DIM), "routed job supported");
        jobs[k] = j;
    }
    TnMoEExpertJob sj = { sg, su, sd, GGUF_TYPE_Q4_K, GGUF_TYPE_Q4_K, GGUF_TYPE_Q4_K, SH, 1.0f };
    TEST_ASSERT(moe_experts_job_ok(&sj, DIM), "shared job supported");
    jobs[TOPK] = sj;
    TnMoEExpertJob bad = sj; bad.down_type = GGUF_TYPE_Q4_K; bad.hdim = EH;  /* 96 % 256 != 0 */
    TEST_ASSERT(!moe_experts_job_ok(&bad, DIM), "K-quant down with hdim % 256 != 0 rejected");

    float out1[DIM], out3[DIM];
    memset(out1, 0, sizeof(out1));
    TEST_ASSERT(moe_experts_forward(out1, x, jobs, TOPK + 1, DIM, 0, NULL) == 0, "fused forward (1 thread)");
    ThreadPool *tp = threadpool_create(3);
    memset(out3, 0, sizeof(out3));
    TEST_ASSERT(moe_experts_forward(out3, x, jobs, TOPK + 1, DIM, 0, tp) == 0, "fused forward (3 threads)");
    threadpool_destroy(tp);
    TEST_ASSERT(memcmp(out1, out3, sizeof(out1)) == 0, "fused result independent of thread count");

    double num = 0, den = 0;
    for (int i = 0; i < DIM; i++) { num += (double)(out1[i] - ref[i]) * (out1[i] - ref[i]); den += (double)ref[i] * ref[i]; }
    double rel = sqrt(num / (den > 0 ? den : 1));
    printf("  fused vs legacy relative L2 error: %.3e (out[0]=%g ref[0]=%g)\n", rel, out1[0], ref[0]);
    TEST_ASSERT(den > 0 && rel < 1e-2, "fused MoE matches the legacy path within activation-quant error");

    /* accumulate semantics: out += result */
    float out_acc[DIM];
    for (int i = 0; i < DIM; i++) out_acc[i] = 1.0f;
    moe_experts_forward(out_acc, x, jobs, TOPK + 1, DIM, 0, NULL);
    int acc_ok = 1;
    for (int i = 0; i < DIM; i++) if (fabsf(out_acc[i] - (out1[i] + 1.0f)) > 1e-4f * (1.0f + fabsf(out1[i]))) acc_ok = 0;
    TEST_ASSERT(acc_ok, "fused forward accumulates into out");

    moe_experts_free();
    free(g); free(u); free(dn); free(sg); free(su); free(sd);
}

int main(void) {
    tn_simd_init();
    printf("=== MoE fused experts / Q8_0-activation kernels ===\n");
    RUN_TEST(test_q8_0_act_dots);
    RUN_TEST(test_q4k_vnni_matches_avx2);
    RUN_TEST(test_fused_matches_legacy);
    TEST_SUMMARY();
}
