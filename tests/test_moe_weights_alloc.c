/*
 * tests/test_moe_weights_alloc.c — moe_weights_alloc / moe_weights_free: which pointer arrays each
 * model kind gets, that free is idempotent, who owns the expert payloads, and that a layer left
 * partly allocated (what a failed allocation leaves behind) frees without crashing or leaking.
 * Leaks are reported by LeakSanitizer in `make test`.
 */
#include <stdlib.h>
#include <string.h>

#include "core/moe_weights.h"
#include "test_harness.h"

#define NL 3
#define NE 4

static void setup(Config *cfg, MoEConfig *mc, TransformerWeights *w, int shared, int mla) {
    memset(cfg, 0, sizeof(*cfg));
    memset(mc, 0, sizeof(*mc));
    memset(w, 0, sizeof(*w));
    cfg->n_layers        = NL;
    mc->is_moe           = 1;
    mc->num_experts      = NE;
    mc->n_shared_experts = shared;
    mc->mla.enabled      = mla;
}

/* 1 when every pointer in the list is non-NULL. */
static int all_set(const void *const *p, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (!p[i]) return 0;
    return 1;
}

static int none_set(const void *const *p, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (p[i]) return 0;
    return 1;
}

#define N(a) (sizeof(a) / sizeof((a)[0]))

static int routed_set(const TransformerWeights *w) {
    const void *const p[] = {w->moe.gate_w, w->moe.gate_s, w->moe.w1, w->moe.w2,
                             w->moe.w3,     w->moe.s1,     w->moe.s2, w->moe.s3};
    return all_set(p, N(p));
}

static int shared_set(const TransformerWeights *w) {
    const void *const p[] = {w->shared.w1, w->shared.w2, w->shared.w3,
                             w->shared.s1, w->shared.s2, w->shared.s3};
    return all_set(p, N(p));
}

static int mla_set(const TransformerWeights *w) {
    const void *const p[] = {w->mla.wq, w->mla.wkv_a, w->mla.wkv_b,
                             w->mla.sq, w->mla.skv_a, w->mla.skv_b};
    return all_set(p, N(p));
}

static int all_null(const TransformerWeights *w) {
    const void *const p[] = {w->moe.gate_w, w->moe.gate_s, w->moe.w1,    w->moe.w2,
                             w->moe.w3,     w->moe.s1,     w->moe.s2,    w->moe.s3,
                             w->shared.w1,  w->shared.w2,  w->shared.w3, w->shared.s1,
                             w->shared.s2,  w->shared.s3,  w->mla.wq,    w->mla.wkv_a,
                             w->mla.wkv_b,  w->mla.sq,     w->mla.skv_a, w->mla.skv_b};
    return none_set(p, N(p));
}

/* Every layer has zeroed per-expert rows. */
static int rows_ok(const TransformerWeights *w) {
    int ok = 1;
    for (int l = 0; l < NL; l++) {
        const void *const rows[] = {w->moe.w1[l], w->moe.w2[l], w->moe.w3[l],
                                    w->moe.s1[l], w->moe.s2[l], w->moe.s3[l]};
        ok &=
            all_set(rows, N(rows)) && w->moe.w1[l][NE - 1] == NULL && w->moe.s3[l][NE - 1] == 0.0f;
    }
    return ok;
}

static void fill_payloads(TransformerWeights *w) {
    for (int l = 0; l < NL; l++)
        for (int e = 0; e < NE; e++) {
            w->moe.w1[l][e] = (tn_i8 *)malloc(8);
            w->moe.w2[l][e] = (tn_i8 *)malloc(8);
            w->moe.w3[l][e] = (tn_i8 *)malloc(8);
        }
}

static void test_dense_allocates_nothing(void) {
    Config cfg;
    MoEConfig mc;
    TransformerWeights w;
    setup(&cfg, &mc, &w, 0, 0);
    mc.is_moe = 0;
    TEST_ASSERT(moe_weights_alloc(&w, &cfg, &mc) == TN_OK, "dense ok");
    TEST_ASSERT(all_null(&w), "no arrays for a dense model");
    moe_weights_free(&w, &mc);
}

static void test_moe_arrays_and_rows(void) {
    Config cfg;
    MoEConfig mc;
    TransformerWeights w;
    setup(&cfg, &mc, &w, 0, 0);
    TEST_ASSERT(moe_weights_alloc(&w, &cfg, &mc) == TN_OK, "moe ok");
    TEST_ASSERT(routed_set(&w), "routed arrays");
    TEST_ASSERT(!w.shared.w1 && !w.mla.wq, "no shared or MLA arrays");
    TEST_ASSERT(w.moe.alloc_layers == NL, "layer count recorded");
    TEST_ASSERT(rows_ok(&w), "every layer has zeroed per-expert rows");
    fill_payloads(&w);
    moe_weights_free(&w, &mc);
    TEST_ASSERT(all_null(&w) && w.moe.alloc_layers == 0, "free nulls everything");
    moe_weights_free(&w, &mc); /* idempotent */
    TEST_ASSERT(all_null(&w), "second free is a no-op");
}

static void test_shared_and_mla(void) {
    Config cfg;
    MoEConfig mc;
    TransformerWeights w;
    setup(&cfg, &mc, &w, 2, 1);
    TEST_ASSERT(moe_weights_alloc(&w, &cfg, &mc) == TN_OK, "moe+shared+mla ok");
    TEST_ASSERT(shared_set(&w), "shared arrays");
    TEST_ASSERT(mla_set(&w), "MLA arrays");
    fill_payloads(&w);
    moe_weights_free(&w, &mc);
    TEST_ASSERT(all_null(&w), "free releases the shared and MLA arrays too");
}

static void test_quantised_experts_own_their_payloads(void) {
    Config cfg;
    MoEConfig mc;
    TransformerWeights w;
    setup(&cfg, &mc, &w, 0, 0);
    TEST_ASSERT(moe_weights_alloc(&w, &cfg, &mc) == TN_OK, "alloc");
    w.expert_quant.present = 1;
    tn_i8 *mine            = (tn_i8 *)malloc(8);
    w.moe.w1[1][2]         = mine;
    moe_weights_free(&w, &mc);
    TEST_ASSERT(all_null(&w), "arrays freed");
    free(mine); /* not freed by moe_weights_free: no double free here */
}

static void test_partly_allocated_layers(void) {
    Config cfg;
    MoEConfig mc;
    TransformerWeights w;
    setup(&cfg, &mc, &w, 0, 0);
    TEST_ASSERT(moe_weights_alloc(&w, &cfg, &mc) == TN_OK, "alloc");
    fill_payloads(&w);
    /* A layer whose w2 row is missing still frees its other rows. */
    for (int e = 0; e < NE; e++) free(w.moe.w2[1][e]);
    free(w.moe.w2[1]);
    w.moe.w2[1] = NULL;
    /* A whole row array that never got allocated must not be dereferenced. */
    for (int l = 0; l < NL; l++) {
        for (int e = 0; e < NE; e++) free(w.moe.w3[l][e]);
        free(w.moe.w3[l]);
    }
    free(w.moe.w3);
    w.moe.w3 = NULL;
    moe_weights_free(&w, &mc);
    TEST_ASSERT(all_null(&w), "partial state frees cleanly");
}

int main(void) {
    printf("=== moe weights alloc/free ===\n");
    RUN_TEST(test_dense_allocates_nothing);
    RUN_TEST(test_moe_arrays_and_rows);
    RUN_TEST(test_shared_and_mla);
    RUN_TEST(test_quantised_experts_own_their_payloads);
    RUN_TEST(test_partly_allocated_layers);
    TEST_SUMMARY();
}
