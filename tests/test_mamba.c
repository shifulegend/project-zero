/*
 * tests/test_mamba.c — Phase 22-SSM Mamba block on a tiny synthetic model:
 * state reset at pos 0, recurrence actually carries history, conv/scan
 * against a scalar re-derivation, and allocation guards.
 */
#include "test_harness.h"
#include "transformer/mamba.h"
#include "core/weights.h"
#include "math/simd_dispatch.h"
#include "sampling/rng.h"

#define DIM 8
#define DI 6
#define DS 3
#define DC 4
#define DR 2

typedef struct {
    float norm[DIM], in[2 * DI * DIM], cw[DI * DC], cb[DI], xw[(DR + 2 * DS) * DI];
    float dtw[DI * DR], dtb[DI], A[DI * DS], D[DI], out[DIM * DI];
} Tiny;

static void tiny_fill(Tiny *t, unsigned long long seed) {
    unsigned long long r; rng_seed(&r, seed);
    float *all = (float *)t;
    for (size_t i = 0; i < sizeof(*t) / sizeof(float); i++) all[i] = (rng_float(&r) - 0.5f) * 0.6f;
    for (int i = 0; i < DIM; i++) t->norm[i] = 1.0f;
    for (int i = 0; i < DI * DS; i++) t->A[i] = -(0.5f + (float)(i % DS)); /* stable decay */
}

static MoEConfig tiny_mc(void) {
    MoEConfig mc; memset(&mc, 0, sizeof(mc));
    mc.is_ssm = 1; mc.ssm_inner_size = DI; mc.ssm_state_size = DS;
    mc.ssm_conv_kernel = DC; mc.ssm_time_step_rank = DR;
    return mc;
}

static void bind(MambaWeights *mw, Tiny *t) {
    mw->norm[0] = t->norm; mw->conv_w[0] = t->cw; mw->conv_b[0] = t->cb; mw->dt_b[0] = t->dtb;
    mw->A[0] = t->A; mw->D[0] = t->D;
    mw->in_proj[0] = (tn_i8 *)t->in;   mw->in_type[0] = WEIGHT_TYPE_F32;
    mw->x_proj[0] = (tn_i8 *)t->xw;    mw->x_type[0] = WEIGHT_TYPE_F32;
    mw->dt_proj[0] = (tn_i8 *)t->dtw;  mw->dt_type[0] = WEIGHT_TYPE_F32;
    mw->out_proj[0] = (tn_i8 *)t->out; mw->out_type[0] = WEIGHT_TYPE_F32;
}

static void run(MambaState *ms, MambaWeights *mw, const Config *cfg, const MoEConfig *mc,
                const float *inputs, int n, float *last) {
    float x[DIM], xb[DIM], xb2[DIM];
    for (int p = 0; p < n; p++) {
        memcpy(x, inputs + p * DIM, sizeof(x));
        mamba_block_forward(x, xb, xb2, ms, mw, cfg, mc, 0, p, NULL);
    }
    memcpy(last, x, sizeof(x));
}

/* Scalar re-derivation of one block step (no SIMD, no dispatch). */
static void ref_step(const Tiny *t, float *cs, float *ss, float *x) {
    float h[DIM], ms2 = 0;
    for (int i = 0; i < DIM; i++) ms2 += x[i] * x[i];
    float inv = 1.0f / sqrtf(ms2 / DIM + 1e-5f);
    for (int i = 0; i < DIM; i++) h[i] = x[i] * inv * t->norm[i];
    float xz[2 * DI];
    for (int r = 0; r < 2 * DI; r++) { xz[r] = 0; for (int c = 0; c < DIM; c++) xz[r] += t->in[r * DIM + c] * h[c]; }
    float uc[DI];
    for (int c = 0; c < DI; c++) {
        float acc = t->cb[c];
        for (int k = 0; k < DC - 1; k++) acc += t->cw[c * DC + k] * cs[c * (DC - 1) + k];
        acc += t->cw[c * DC + DC - 1] * xz[c];
        for (int k = 0; k < DC - 2; k++) cs[c * (DC - 1) + k] = cs[c * (DC - 1) + k + 1];
        cs[c * (DC - 1) + DC - 2] = xz[c];
        uc[c] = acc / (1 + expf(-acc));
    }
    float xdb[DR + 2 * DS];
    for (int r = 0; r < DR + 2 * DS; r++) { xdb[r] = 0; for (int c = 0; c < DI; c++) xdb[r] += t->xw[r * DI + c] * uc[c]; }
    float y[DI];
    for (int c = 0; c < DI; c++) {
        float dt = t->dtb[c];
        for (int k = 0; k < DR; k++) dt += t->dtw[c * DR + k] * xdb[k];
        dt = dt > 20 ? dt : log1pf(expf(dt));
        float acc = 0;
        for (int n = 0; n < DS; n++) {
            float *s = &ss[c * DS + n];
            *s = *s * expf(dt * t->A[c * DS + n]) + dt * xdb[DR + n] * uc[c];
            acc += *s * xdb[DR + DS + n];
        }
        float z = xz[DI + c];
        y[c] = (acc + t->D[c] * uc[c]) * (z / (1 + expf(-z)));
    }
    for (int i = 0; i < DIM; i++) { float o = 0; for (int c = 0; c < DI; c++) o += t->out[i * DI + c] * y[c]; x[i] += o; }
}

static void test_matches_scalar_reference(void) {
    Tiny t; tiny_fill(&t, 11);
    Config cfg; memset(&cfg, 0, sizeof(cfg)); cfg.dim = DIM; cfg.n_layers = 1; cfg.rms_norm_eps = 1e-5f;
    MoEConfig mc = tiny_mc();
    MambaWeights *mw = NULL; MambaState *ms = NULL;
    TEST_ASSERT_EQ(mamba_weights_alloc(&mw, 1), TN_OK, "weights alloc");
    TEST_ASSERT_EQ(mamba_state_alloc(&ms, 1, &mc), TN_OK, "state alloc");
    bind(mw, &t);
    float in[5 * DIM]; unsigned long long r; rng_seed(&r, 5);
    for (int i = 0; i < 5 * DIM; i++) in[i] = rng_float(&r) - 0.5f;
    float got[DIM];
    run(ms, mw, &cfg, &mc, in, 5, got);
    float cs[DI * (DC - 1)] = {0}, ss[DI * DS] = {0}, want[DIM];
    for (int p = 0; p < 5; p++) { memcpy(want, in + p * DIM, sizeof(want)); ref_step(&t, cs, ss, want); }
    float md = 0;
    for (int i = 0; i < DIM; i++) { float e = fabsf(got[i] - want[i]); if (e > md) md = e; }
    TEST_ASSERT(md < 1e-4f, "5-step output matches the scalar re-derivation");

    /* Recurrence carries history: same final input, different prefix -> different output. */
    float in2[5 * DIM]; memcpy(in2, in, sizeof(in2)); in2[0] += 1.0f;
    float got2[DIM]; run(ms, mw, &cfg, &mc, in2, 5, got2);
    int differs = 0; for (int i = 0; i < DIM; i++) differs |= fabsf(got2[i] - got[i]) > 1e-6f;
    TEST_ASSERT(differs, "earlier tokens influence later outputs");

    /* pos 0 resets: re-running the original sequence reproduces it exactly. */
    float got3[DIM]; run(ms, mw, &cfg, &mc, in, 5, got3);
    int same = 1; for (int i = 0; i < DIM; i++) same &= got3[i] == got[i];
    TEST_ASSERT(same, "state reset at pos 0 makes runs reproducible");
    mamba_state_free(ms); mamba_weights_free(mw);
}

static void test_alloc_guards(void) {
    MoEConfig mc; memset(&mc, 0, sizeof(mc));
    MambaState *ms = NULL; MambaWeights *mw = NULL;
    TEST_ASSERT_EQ(mamba_state_alloc(&ms, 1, &mc), TN_ERR_INVALID_ARGS, "missing ssm dims rejected");
    TEST_ASSERT_EQ(mamba_weights_alloc(&mw, 0), TN_ERR_INVALID_ARGS, "0 layers rejected");
    TEST_ASSERT(moe_is_recurrent(&(MoEConfig){ .is_ssm = 1 }), "SSM is recurrent");
    TEST_ASSERT(!moe_is_recurrent(NULL), "NULL = dense");
    mamba_state_free(NULL); mamba_weights_free(NULL);
}

int main(void) {
    tn_simd_init();
    printf("=== Mamba SSM (Phase 22-SSM) Tests ===\n");
    RUN_TEST(test_matches_scalar_reference);
    RUN_TEST(test_alloc_guards);
    TEST_SUMMARY();
}
