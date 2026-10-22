/*
 * tests/test_batch_scheduler.c — Phase 23 stage 2b (2026-10-22): requests
 * served together by the parallel scheduler produce the same text as the
 * same requests run one at a time through generate_ex(); fewer slots than
 * requests queue, a tight KV budget makes requests wait / evicts idle slots,
 * and a callback that stops early cancels only its own request.
 *
 * Needs a real dense GGUF (SmolLM2): $PZ_TEST_MODEL, else
 * models/smollm2-q8_0.gguf; skipped when absent.
 */
#include "test_harness.h"
#include "api/batch_scheduler.h"
#include "cli/model_load.h"
#include "threading/thread_pool.h"
#include "math/simd_dispatch.h"
#include "tokenizer/tokenizer_gguf.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_REQ 4
#define MAX_TOK 20

static LoadedModel M;
static Tokenizer T;
static ThreadPool *TP;

static const char *PROMPTS[N_REQ] = {
    "What is the capital of France?",
    "Write one sentence about the sea.",
    "List three colours.",
    "Explain gravity to a child in one sentence.",
};

typedef struct { char text[4096]; size_t len; int pieces; int stop_after; } Out;

static int collect(const char *piece, void *ud) {
    Out *o = (Out *)ud;
    size_t n = strlen(piece);
    if (o->len + n < sizeof(o->text)) { memcpy(o->text + o->len, piece, n + 1); o->len += n; }
    o->pieces++;
    return o->stop_after > 0 && o->pieces >= o->stop_after;
}

static void params(GenerationParams *gp, int i, float temp, GenerationStats *st) {
    gen_params_default(gp);
    gp->max_tokens = MAX_TOK;
    gp->sampler.temperature = temp;
    gp->seed = 100 + i;
    gp->stats = st;
}

static void sequential(float temp, Out *out) {
    for (int i = 0; i < N_REQ; i++) {
        GenerationParams gp;
        GenerationStats st;
        params(&gp, i, temp, &st);
        memset(&out[i], 0, sizeof(out[i]));
        M.state->current_pos = 0;
        generate_ex(&M.config, &M.weights, M.state, &M.moe_config, &T, TP, PROMPTS[i], &gp,
                    collect, &out[i]);
    }
}

typedef struct {
    BatchScheduler *bs; int i; float temp; int use_cache; Out out; GenerationStats st;
} Req;

static void *run_req(void *arg) {
    Req *r = (Req *)arg;
    GenerationParams gp;
    params(&gp, r->i, r->temp, &r->st);
    batch_scheduler_generate(r->bs, PROMPTS[r->i], &gp, r->use_cache, collect, &r->out);
    return NULL;
}

static void parallel(BatchScheduler *bs, float temp, int use_cache, Req *req, int stop_req, int stop_after) {
    pthread_t th[N_REQ];
    for (int i = 0; i < N_REQ; i++) {
        memset(&req[i], 0, sizeof(req[i]));
        req[i].bs = bs; req[i].i = i; req[i].temp = temp; req[i].use_cache = use_cache;
        if (i == stop_req) req[i].out.stop_after = stop_after;
        pthread_create(&th[i], NULL, run_req, &req[i]);
    }
    for (int i = 0; i < N_REQ; i++) pthread_join(th[i], NULL);
}

static BatchScheduler *make(int slots, int rows, long long budget, PrefixCache *pc) {
    BatchSchedulerOptions o = { slots, rows, budget };
    memset(pc, 0, sizeof(*pc));
    return batch_scheduler_create(&M.config, &M.weights, M.state, pc, NULL, &M.moe_config,
                                  &T, TP, &o);
}

static int same(const Out *seq, const Req *par, const char *what) {
    int ok = 1;
    for (int i = 0; i < N_REQ; i++) {
        if (strcmp(seq[i].text, par[i].out.text) != 0) {
            printf("  %s: request %d differs\n    seq: %s\n    par: %s\n", what, i,
                   seq[i].text, par[i].out.text);
            ok = 0;
        }
    }
    return ok;
}

static void test_matches_sequential(void) {
    Out seq[N_REQ];
    Req par[N_REQ];
    PrefixCache pc;
    BatchScheduler *bs = make(N_REQ, 64, 0, &pc);
    TEST_ASSERT(bs != NULL, "scheduler with 4 slots");
    if (!bs) return;
    for (int t = 0; t < 2; t++) {
        float temp = t == 0 ? 0.0f : 0.8f;
        sequential(temp, seq);
        parallel(bs, temp, 0, par, -1, 0);
        TEST_ASSERT(same(seq, par, temp == 0.0f ? "greedy" : "seeded 0.8"),
                    temp == 0.0f ? "greedy: parallel text == sequential text"
                                 : "seeded sampling: parallel text == sequential text");
        int stats_ok = 1;
        for (int i = 0; i < N_REQ; i++)
            if (par[i].st.completion_tokens <= 0 || par[i].st.prompt_tokens <= 0) stats_ok = 0;
        TEST_ASSERT(stats_ok, "every request reports prompt/completion token counts");
    }
    /* Prefix cache on: the second round reuses each prompt's cached K/V. */
    parallel(bs, 0.0f, 1, par, -1, 0);
    parallel(bs, 0.0f, 1, par, -1, 0);
    int cached = 0;
    for (int i = 0; i < N_REQ; i++) cached += par[i].st.cached_tokens > 0;
    sequential(0.0f, seq);
    TEST_ASSERT(same(seq, par, "cached"), "with prefix cache: text unchanged");
    TEST_ASSERT(cached == N_REQ, "repeated prompts reuse their slot's cached prefix");
    batch_scheduler_destroy(bs);
    prefix_cache_free(&pc);
}

static void test_queue_budget_cancel(void) {
    Out seq[N_REQ];
    Req par[N_REQ];
    PrefixCache pc;
    /* Two slots for four requests, and a budget that fits about one request:
     * requests must wait for room, idle slots give their KV back. */
    BatchScheduler *bs = make(2, 8, 64, &pc);
    TEST_ASSERT(bs != NULL, "scheduler with 2 slots, 8-row steps, 64-token budget");
    if (!bs) return;
    sequential(0.0f, seq);
    parallel(bs, 0.0f, 1, par, -1, 0);
    TEST_ASSERT(same(seq, par, "queued"), "queued + budget-limited requests: same text");
    BatchSchedulerStats st;
    batch_scheduler_stats(bs, &st);
    TEST_ASSERT(st.active == 0 && st.pending == 0, "scheduler idle afterwards");
    TEST_ASSERT(st.kv_committed_tokens <= 64 + MAX_TOK + 64, "KV held stays near the budget");

    parallel(bs, 0.0f, 0, par, 1, 3);
    TEST_ASSERT(par[1].out.pieces == 3 && par[1].st.finish_reason == GEN_FINISH_CANCELLED,
                "a stopping callback cancels its request after 3 pieces");
    TEST_ASSERT(strcmp(par[0].out.text, seq[0].text) == 0 && strcmp(par[2].out.text, seq[2].text) == 0,
                "other requests unaffected by the cancel");
    batch_scheduler_destroy(bs);
    prefix_cache_free(&pc);
}

int main(void) {
    printf("=== parallel batch scheduler ===\n");
    const char *path = getenv("PZ_TEST_MODEL");
    if (!path || !*path) path = "models/smollm2-q8_0.gguf";
    FILE *f = fopen(path, "rb");
    if (!f) { printf("  (no model at %s; set PZ_TEST_MODEL) skipped\n", path); return 0; }
    fclose(f);
    tn_simd_init();
    TP = threadpool_create(4);
    memset(&M, 0, sizeof(M));
    if (load_gguf_model(&M, path, true, 0, TP, false) != TN_OK) {
        printf("  (could not load %s) skipped\n", path);
        threadpool_destroy(TP);
        return 0;
    }
    if (!batch_scheduler_supported(&M.weights, &M.moe_config)) {
        printf("  (model not supported by the batched forward) skipped\n");
        loaded_model_free(&M);
        threadpool_destroy(TP);
        return 0;
    }
    memset(&T, 0, sizeof(T));
    if (tokenizer_load_from_gguf(&T, &M.gguf_hdr) != TN_OK) {
        printf("  (no tokenizer in %s) skipped\n", path);
        loaded_model_free(&M);
        threadpool_destroy(TP);
        return 0;
    }
    RUN_TEST(test_matches_sequential);
    RUN_TEST(test_queue_budget_cancel);
    tokenizer_free(&T);
    loaded_model_free(&M);
    threadpool_destroy(TP);
    TEST_SUMMARY();
}
