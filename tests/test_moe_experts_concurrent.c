/*
 * test_moe_experts_concurrent.c -- moe_experts_forward() from two threads at once, with different
 * model widths, must give each thread the single-thread result. The executor's scratch used to be
 * file-scope statics, so concurrent callers overwrote and reallocated each other's buffers (TSan
 * and the comparison below fail on that code).
 */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/gguf_reader.h"
#include "math/simd_dispatch.h"
#include "transformer/moe_experts.h"

static int g_fail;
#define CHECK(c, msg)                                                                              \
    do {                                                                                           \
        if (!(c)) {                                                                                \
            printf("FAIL: %s\n", msg);                                                             \
            g_fail++;                                                                              \
        } else                                                                                     \
            printf("ok: %s\n", msg);                                                               \
    } while (0)

#define MAX_JOBS 4

typedef struct {
    int dim, hdim, n_jobs;
    float *x;
    uint8_t *w[MAX_JOBS][3]; /* gate, up, down */
    TnMoEExpertJob jobs[MAX_JOBS];
    float ref[1024];
} Model;

static uint8_t *make_q4k(int rows, int cols, unsigned *seed) {
    size_t n   = (size_t)rows * (size_t)(cols / 256) * 144;
    uint8_t *w = (uint8_t *)malloc(n);
    for (size_t i = 0; i < n; i++) {
        *seed = *seed * 1664525u + 1013904223u;
        w[i]  = (uint8_t)(*seed >> 24);
    }
    for (size_t b = 0; b < n / 144; b++) { /* d = dmin = fp16 2^-14: keeps silu out of saturation */
        w[b * 144 + 0] = 0x00;
        w[b * 144 + 1] = 0x04;
        w[b * 144 + 2] = 0x00;
        w[b * 144 + 3] = 0x04;
    }
    return w;
}

static void build(Model *m, int dim, int hdim, int n_jobs, unsigned seed) {
    memset(m, 0, sizeof(*m));
    m->dim    = dim;
    m->hdim   = hdim;
    m->n_jobs = n_jobs;
    m->x      = (float *)malloc((size_t)dim * sizeof(float));
    for (int i = 0; i < dim; i++) {
        seed    = seed * 1664525u + 1013904223u;
        m->x[i] = ((int)(seed >> 24) - 128) / 128.0f;
    }
    for (int j = 0; j < n_jobs; j++) {
        m->w[j][0]        = make_q4k(hdim, dim, &seed);
        m->w[j][1]        = make_q4k(hdim, dim, &seed);
        m->w[j][2]        = make_q4k(dim, hdim, &seed);
        TnMoEExpertJob jb = {m->w[j][0],     m->w[j][1],     m->w[j][2], GGUF_TYPE_Q4_K,
                             GGUF_TYPE_Q4_K, GGUF_TYPE_Q4_K, hdim,       0.25f * (float)(j + 1)};
        m->jobs[j]        = jb;
    }
}

static void destroy(Model *m) {
    free(m->x);
    for (int j = 0; j < m->n_jobs; j++)
        for (int k = 0; k < 3; k++) free(m->w[j][k]);
}

typedef struct {
    Model *a, *b;
    int ok;
} Job;

static void *run_fn(void *arg) {
    Job *j = (Job *)arg;
    j->ok  = 1;
    for (int rep = 0; rep < 200; rep++) {
        Model *m = (rep & 1) ? j->b : j->a; /* alternate sizes so the scratch has to regrow */
        float out[1024];
        memset(out, 0, sizeof(out));
        if (moe_experts_forward(out, m->x, m->jobs, m->n_jobs, m->dim, 0, NULL) != 0 ||
            memcmp(out, m->ref, (size_t)m->dim * sizeof(float)) != 0)
            j->ok = 0;
    }
    return NULL;
}

int main(void) {
    tn_simd_init();
    Model small, big;
    build(&small, 256, 256, 2, 5);
    build(&big, 512, 512, 4, 6);
    for (Model *m = &small; m; m = (m == &small) ? &big : NULL) {
        float out[1024];
        memset(out, 0, sizeof(out));
        CHECK(moe_experts_forward(out, m->x, m->jobs, m->n_jobs, m->dim, 0, NULL) == 0,
              "single-thread reference run");
        double mag = 0;
        for (int i = 0; i < m->dim; i++) mag += out[i] < 0 ? -out[i] : out[i];
        CHECK(mag > 1e-3, "the reference output is not trivially zero");
        memcpy(m->ref, out, sizeof(out));
    }

    Job a = {&small, &big, 0}, b = {&big, &small, 0}; /* opposite phase */
    pthread_t ta, tb;
    pthread_create(&ta, NULL, run_fn, &a);
    pthread_create(&tb, NULL, run_fn, &b);
    pthread_join(ta, NULL);
    pthread_join(tb, NULL);
    CHECK(a.ok && b.ok, "two threads reproduce the single-thread MoE expert output exactly");

    moe_experts_free();
    destroy(&small);
    destroy(&big);
    printf("=== %s ===\n", g_fail ? "FAILED" : "moe experts concurrent tests passed");
    return g_fail ? 1 : 0;
}
