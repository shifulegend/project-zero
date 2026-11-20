/*
 * test_qwen3moe_concurrent.c -- two threads run Qwen3-MoE forward passes on separate RunStates at
 * once. Run under TSan (make test-tsan) it reports any scratch buffer the two share.
 */
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "core/config.h"
#include "core/lora.h"
#include "core/moe_config.h"
#include "core/moe_weights.h"
#include "core/unpack.h"
#include "core/weights.h"
#include "math/simd_dispatch.h"
#include "runtime/run_state.h"
#include "runtime/spec_scratch.h"
#include "test_harness.h"
#include "transformer/forward.h"

#define TOL 1e-3f

#include "synth_models.h"

/* Two threads run the same Qwen3-MoE token sequence on their own RunStates at once. The attention
 * scratch used to be function-local statics shared by every caller, so concurrent forwards
 * overwrote each other's Q/K/V (2026-11-19). Each thread must reproduce the single-thread logits
 * exactly; under TSan (make test-tsan) any remaining shared buffer is reported as a race. */
typedef struct {
    const Config *cfg;
    const TransformerWeights *w;
    const MoEConfig *mc;
    float out[3][Q3_VOCAB];
    int ok;
} Q3Job;

static void *q3_job_fn(void *arg) {
    Q3Job *j      = (Q3Job *)arg;
    int tokens[3] = {1, 3, 2};
    RunState s;
    j->ok = 0;
    if (run_state_alloc_ex(&s, j->cfg, j->cfg->seq_len, true, NULL) != TN_OK) return NULL;
    if (qwen3moe_run_state_alloc(&s, j->cfg, j->mc, j->cfg->seq_len) != TN_OK) {
        run_state_free(&s);
        return NULL;
    }
    sw_init(&s.sw, j->cfg->seq_len, 0);
    for (int rep = 0; rep < 20; rep++) { /* repeat to widen the overlap window */
        s.current_pos = 0;
        sw_init(&s.sw, j->cfg->seq_len, 0);
        for (int k = 0; k < 3; k++) {
            float *lg = transformer_forward(tokens[k], k, j->cfg, j->w, &s, j->mc, NULL);
            memcpy(j->out[k], lg, Q3_VOCAB * sizeof(float));
        }
    }
    j->ok = 1;
    qwen3moe_run_state_free(&s, j->cfg);
    run_state_free(&s);
    return NULL;
}

static void test_qwen3moe_concurrent_runstates(void) {
    tn_simd_init();
    Config cfg;
    TransformerWeights w;
    MoEConfig mc;
    build_qwen3moe_ternary_model(&cfg, &w, &mc);

    Q3Job ref = {&cfg, &w, &mc, {{0}}, 0};
    q3_job_fn(&ref);
    TEST_ASSERT(ref.ok, "Q3MoE single-thread reference run");

    Q3Job a = {&cfg, &w, &mc, {{0}}, 0}, b = {&cfg, &w, &mc, {{0}}, 0};
    pthread_t ta, tb;
    pthread_create(&ta, NULL, q3_job_fn, &a);
    pthread_create(&tb, NULL, q3_job_fn, &b);
    pthread_join(ta, NULL);
    pthread_join(tb, NULL);
    TEST_ASSERT(a.ok && b.ok, "Q3MoE concurrent runs completed");
    TEST_ASSERT(memcmp(ref.out, a.out, sizeof(ref.out)) == 0 &&
                    memcmp(ref.out, b.out, sizeof(ref.out)) == 0,
                "two threads on separate RunStates reproduce the single-thread logits exactly");
    free_qwen3moe_ternary_model(&w);
}

int main(void) {
    setenv("TN_FORCE_BACKEND", "avx2", 1);
    RUN_TEST(test_qwen3moe_concurrent_runstates);
    TEST_SUMMARY();
}
