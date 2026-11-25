/*
 * test_tls_scratch.c -- tn_tls_scratch() gives each thread its own growable buffer, and the
 * Q4_K matmul that uses it gives the same answer when two threads call it at once with different
 * sizes. (The matmul used a shared static buffer that one thread could free while the other read
 * it; under TSan/ASan the old code fails this test.)
 */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "math/matmul_q4k.h"
#include "memory/tls_scratch.h"

static int g_fail;
#define CHECK(c, msg)                                                                              \
    do {                                                                                           \
        if (!(c)) {                                                                                \
            printf("FAIL: %s\n", msg);                                                             \
            g_fail++;                                                                              \
        } else                                                                                     \
            printf("ok: %s\n", msg);                                                               \
    } while (0)

static TnTlsScratch g_scratch = TN_TLS_SCRATCH_INIT;

static void test_basic(void) {
    unsigned char *a = (unsigned char *)tn_tls_scratch(&g_scratch, 100);
    CHECK(a != NULL, "first call returns a buffer");
    memset(a, 0xAB, 100);
    unsigned char *b = (unsigned char *)tn_tls_scratch(&g_scratch, 50);
    CHECK(b == a && b[0] == 0xAB && b[49] == 0xAB, "a smaller request reuses the buffer unchanged");
    unsigned char *c = (unsigned char *)tn_tls_scratch(&g_scratch, 1u << 20);
    CHECK(c != NULL, "growing works");
    memset(c, 1, 1u << 20);
    CHECK(tn_tls_scratch(&g_scratch, 0) != NULL, "zero bytes still returns a pointer");
    CHECK(tn_tls_scratch(NULL, 8) == NULL, "NULL scratch object returns NULL");
}

typedef struct {
    int id;
    int ok;
} Job;

/* Each thread grows and fills its own buffer repeatedly; a shared buffer would show the other
 * thread's pattern. */
static void *isolation_fn(void *arg) {
    Job *j = (Job *)arg;
    j->ok  = 1;
    for (int i = 1; i <= 400; i++) {
        size_t n         = (size_t)i * 64;
        unsigned char *p = (unsigned char *)tn_tls_scratch(&g_scratch, n);
        if (!p) {
            j->ok = 0;
            return NULL;
        }
        memset(p, 0x10 + j->id, n);
        for (size_t k = 0; k < n; k += 61)
            if (p[k] != (unsigned char)(0x10 + j->id)) j->ok = 0;
    }
    return NULL;
}

static void test_isolation(void) {
    Job a = {1, 0}, b = {2, 0};
    pthread_t ta, tb;
    pthread_create(&ta, NULL, isolation_fn, &a);
    pthread_create(&tb, NULL, isolation_fn, &b);
    pthread_join(ta, NULL);
    pthread_join(tb, NULL);
    CHECK(a.ok && b.ok, "two threads growing their buffers never see each other's data");
}

/* ── Q4_K matmul from two threads at once ─────────────────────────────── */

#define SMALL_N 256
#define BIG_N 2048
#define ROWS 8

typedef struct {
    int n;
    uint8_t *w;
    float *x;
    float ref[ROWS];
} Case;

static void fill_case(Case *c, int n, unsigned seed) {
    int nb = n / 256;
    c->n   = n;
    c->w   = (uint8_t *)malloc((size_t)ROWS * nb * 144);
    c->x   = (float *)malloc((size_t)n * sizeof(float));
    for (size_t i = 0; i < (size_t)ROWS * nb * 144; i++) {
        seed    = seed * 1664525u + 1013904223u;
        c->w[i] = (uint8_t)(seed >> 24);
    }
    for (size_t blk = 0; blk < (size_t)ROWS * nb; blk++) { /* d, dmin = fp16 1.0 and 0.5 */
        c->w[blk * 144 + 0] = 0x00;
        c->w[blk * 144 + 1] = 0x3C;
        c->w[blk * 144 + 2] = 0x00;
        c->w[blk * 144 + 3] = 0x38;
    }
    for (int i = 0; i < n; i++) {
        seed    = seed * 1664525u + 1013904223u;
        c->x[i] = ((int)(seed >> 24) - 128) / 64.0f;
    }
}

typedef struct {
    Case *small, *big;
    int ok;
} MatmulJob;

static void *matmul_fn(void *arg) {
    MatmulJob *j = (MatmulJob *)arg;
    j->ok        = 1;
    for (int rep = 0; rep < 300; rep++) {
        Case *c = (rep & 1) ? j->big : j->small; /* alternate sizes: forces buffer growth */
        float out[ROWS];
        parallel_matmul_q4k(out, c->x, c->w, c->n, ROWS, NULL);
        if (memcmp(out, c->ref, sizeof(out)) != 0) j->ok = 0;
    }
    return NULL;
}

static void test_matmul_two_threads(void) {
    Case small, big;
    fill_case(&small, SMALL_N, 7);
    fill_case(&big, BIG_N, 9);
    parallel_matmul_q4k(small.ref, small.x, small.w, small.n, ROWS, NULL);
    parallel_matmul_q4k(big.ref, big.x, big.w, big.n, ROWS, NULL);

    MatmulJob a = {&small, &big, 0}, b = {&big, &small, 0}; /* opposite phase */
    pthread_t ta, tb;
    pthread_create(&ta, NULL, matmul_fn, &a);
    pthread_create(&tb, NULL, matmul_fn, &b);
    pthread_join(ta, NULL);
    pthread_join(tb, NULL);
    CHECK(a.ok && b.ok, "Q4_K matmul from two threads matches the single-thread result");
    free(small.w);
    free(small.x);
    free(big.w);
    free(big.x);
}

int main(void) {
    test_basic();
    test_isolation();
    test_matmul_two_threads();
    printf("=== %s ===\n", g_fail ? "FAILED" : "tls scratch tests passed");
    return g_fail ? 1 : 0;
}
