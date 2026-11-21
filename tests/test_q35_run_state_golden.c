/*
 * tests/test_q35_run_state_golden.c — q35_run_state_alloc over 3,000 generated model shapes:
 * valid and invalid layer counts, head sizes, convolution kernels and state sizes, partial-rotary
 * dims, attention intervals, and sequence lengths long enough to hit the 600 MB context ceiling.
 * For each: the result code, the context length left in the run state, which layers hold a K/V
 * cache and which hold convolution and recurrent state, and the RoPE table; the last element of
 * every buffer is written under ASan, so any buffer smaller than the attention code needs fails
 * loudly. Hash recorded from the function before it was split into steps.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "runtime/run_state.h"
#include "test_harness.h"

#define EXPECTED_HASH 0x8a0df8733dbf2e6eull
#define N_CASES 3000

static uint64_t g_hash = 0xcbf29ce484222325ull;
static void mix(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        g_hash ^= b[i];
        g_hash *= 0x100000001b3ull;
    }
}

static uint64_t g_rng = 0x2545f4914f6cdd1dull;
static unsigned rnd(unsigned n) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (unsigned)(g_rng % n);
}

/* Context length asked for: empty, negative, tiny, ordinary, or far past the 600 MB ceiling. */
static void gen_seq(int *seq) {
    switch (rnd(8)) {
        case 0:
            *seq = 0;
            break;
        case 1:
            *seq = -5;
            break;
        case 2:
            *seq = 1;
            break;
        case 3:
            *seq = 5000000;
            break; /* far past the context ceiling */
        default:
            *seq = 1 + (int)rnd(3000);
            break;
    }
}

/* A model shape; some fields are drawn invalid on purpose. */
static void gen_shape(Config *cfg, MoEConfig *mc, int *seq) {
    memset(cfg, 0, sizeof *cfg);
    memset(mc, 0, sizeof *mc);
    cfg->n_layers               = (int)rnd(8) - (rnd(20) == 0 ? 1 : 0);
    cfg->n_heads                = 1 + (int)rnd(4);
    cfg->n_kv_heads             = (int)rnd(4) - (rnd(20) == 0 ? 1 : 0);
    cfg->rope_theta             = rnd(3) == 0 ? 0.0f : (rnd(2) ? 10000.0f : 1000000.0f);
    mc->has_linear_attn         = rnd(25) != 0;
    mc->full_attention_interval = (int)rnd(5);
    mc->attn_head_dim           = (int)rnd(40) - (rnd(20) == 0 ? 3 : 0);
    mc->attn_rope_dim           = rnd(3) ? 0 : 2 * (int)rnd(16);
    mc->ssm_conv_kernel         = (int)rnd(6);
    mc->ssm_group_count         = 1 + (int)rnd(3);
    mc->ssm_state_size          = (int)rnd(9) - (rnd(20) == 0 ? 1 : 0);
    mc->ssm_time_step_rank      = (int)rnd(5) - (rnd(20) == 0 ? 1 : 0);
    mc->ssm_inner_size          = mc->ssm_time_step_rank * mc->ssm_state_size;
    gen_seq(seq);
}

/* Writes to the last element of each buffer the attention code expects (sizes from the shape). */
static void touch_buffers(const RunState *s, const Config *cfg, const MoEConfig *mc) {
    size_t n_kv = (size_t)cfg->n_kv_heads, hd = (size_t)mc->attn_head_dim;
    size_t conv_dim = (size_t)mc->ssm_group_count * mc->ssm_state_size * 2 + mc->ssm_inner_size;
    for (int l = 0; l < cfg->n_layers; l++) {
        if (q35_layer_is_full_attn(mc, l)) {
            size_t kv                     = n_kv * (size_t)s->max_seq_len * hd;
            s->q35.key_cache[l][kv - 1]   = 1.0f;
            s->q35.value_cache[l][kv - 1] = 1.0f;
        } else {
            size_t conv  = (size_t)(mc->ssm_conv_kernel - 1) * conv_dim;
            size_t recur = (size_t)mc->ssm_time_step_rank * mc->ssm_state_size * mc->ssm_state_size;
            s->q35.conv_state[l][conv - 1]   = 1.0f;
            s->q35.recur_state[l][recur - 1] = 1.0f;
        }
    }
    size_t q_w = (size_t)cfg->n_heads * hd, kv_w = n_kv * hd;
    size_t lin = conv_dim * 2 + (size_t)mc->ssm_inner_size * 2 + (size_t)mc->ssm_time_step_rank * 3;
    s->q35.scratch[2 * q_w + 2 * kv_w + q_w + lin - 1] = 1.0f;
}

static void one_case(void) {
    Config cfg;
    MoEConfig mc;
    int seq;
    gen_shape(&cfg, &mc, &seq);
    RunState s;
    memset(&s, 0, sizeof s);
    s.max_seq_len = 777; /* what an earlier allocation left; the call must overwrite it */
    int rc        = (int)q35_run_state_alloc(&s, &cfg, &mc, seq);
    mix(&rc, sizeof rc);
    mix(&s.max_seq_len, sizeof s.max_seq_len);
    if (rc == TN_OK && mc.has_linear_attn) {
        touch_buffers(&s, &cfg, &mc);
        for (int l = 0; l < cfg.n_layers; l++) {
            int has[4] = {s.q35.key_cache[l] != NULL, s.q35.value_cache[l] != NULL,
                          s.q35.conv_state[l] != NULL, s.q35.recur_state[l] != NULL};
            mix(has, sizeof has);
        }
        int rope_dim = mc.attn_rope_dim > 0 ? mc.attn_rope_dim : mc.attn_head_dim;
        mix(s.q35.rope_freq, (size_t)(rope_dim / 2) * sizeof(float));
        q35_run_state_free(&s, &cfg, &mc);
    } else if (rc == TN_OK) {
        mix(&s.q35.key_cache, sizeof s.q35.key_cache); /* not a hybrid model: untouched (NULL) */
    }
}

static void test_alloc_golden(void) {
    for (int t = 0; t < N_CASES; t++) one_case();
    if (getenv("Q35_GOLDEN_PRINT")) printf("hash 0x%016llxull\n", (unsigned long long)g_hash);
    TEST_ASSERT(g_hash == EXPECTED_HASH, "run-state layouts match the recorded hash");
}

int main(void) {
    printf("=== q35_run_state_alloc golden ===\n");
    RUN_TEST(test_alloc_golden);
    TEST_SUMMARY();
}
