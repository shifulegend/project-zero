/* sample_top_p(): same results as the previous shared-buffer implementation (kept here as a
 * reference), correct for flat distributions that exceed the stack buffer and for vocabularies
 * above the old 200000 limit, and safe from several threads at once (runs under TSan). */
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include "sampling/rng.h"
#include "sampling/sampling.h"

static int g_fail;
#define CHECK(c, msg)                                                                              \
    do {                                                                                           \
        if (!(c)) {                                                                                \
            printf("FAIL: %s\n", msg);                                                             \
            g_fail++;                                                                              \
        } else                                                                                     \
            printf("ok: %s\n", msg);                                                               \
    } while (0)

typedef struct {
    float prob;
    int index;
} PI;
static int cmp(const void *a, const void *b) {
    float pa = ((const PI *)a)->prob, pb = ((const PI *)b)->prob;
    return pa > pb ? -1 : (pa < pb ? 1 : 0);
}

/* the old algorithm, with a private heap buffer instead of the static one */
static int reference(const float *logits, int n_vocab, float top_p, unsigned long long *rng) {
    float mx = logits[0];
    for (int i = 1; i < n_vocab; i++)
        if (logits[i] > mx) mx = logits[i];
    float thr = mx - 20.0f;
    PI *c     = (PI *)malloc((size_t)n_vocab * sizeof(PI));
    int n     = 0;
    float sum = 0.0f;
    for (int i = 0; i < n_vocab; i++)
        if (logits[i] >= thr) {
            float e    = expf(logits[i] - mx);
            c[n].prob  = e;
            c[n].index = i;
            sum += e;
            n++;
        }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) c[i].prob *= inv;
    qsort(c, (size_t)n, sizeof(PI), cmp);
    float cum = 0.0f;
    int cut   = n;
    for (int i = 0; i < n; i++) {
        cum += c[i].prob;
        if (cum >= top_p) {
            cut = i + 1;
            break;
        }
    }
    float ts = 0.0f;
    for (int i = 0; i < cut; i++) ts += c[i].prob;
    float r = rng_float(rng) * ts, cdf = 0.0f;
    int res = c[0].index;
    for (int i = 0; i < cut; i++) {
        cdf += c[i].prob;
        if (r < cdf) {
            res = c[i].index;
            break;
        }
    }
    free(c);
    return res;
}

static void fill(float *l, int n, unsigned long long *s, float spread) {
    for (int i = 0; i < n; i++) {
        *s   = *s * 6364136223846793005ULL + 1442695040888963407ULL;
        l[i] = ((float)((*s >> 33) & 0xFFFF) / 65535.0f) * spread;
    }
}

typedef struct {
    int ok;
    unsigned long long seed;
} Job;
static void *worker(void *arg) {
    Job *j   = (Job *)arg;
    j->ok    = 1;
    float *l = (float *)malloc(5000 * sizeof(float));
    for (int it = 0; it < 300; it++) {
        unsigned long long s = j->seed + (unsigned long long)it, r1 = s * 7 + 1, r2 = r1;
        fill(l, 5000, &s, 6.0f);
        if (sample_top_p(l, 5000, 0.9f, &r1) != reference(l, 5000, 0.9f, &r2)) j->ok = 0;
    }
    free(l);
    return NULL;
}

int main(void) {
    unsigned long long s = 99;
    int same             = 1;
    float *l             = (float *)malloc(60000 * sizeof(float));
    for (int it = 0; it < 400; it++) {
        int n = (it % 3 == 0) ? 50 : (it % 3 == 1 ? 800 : 5000); /* few candidates ... many */
        float spread = (it % 2) ? 3.0f : 25.0f;
        fill(l, n, &s, spread);
        unsigned long long r1 = 1000 + (unsigned long long)it, r2 = r1;
        float tp = 0.3f + 0.7f * (float)(it % 10) / 10.0f;
        if (sample_top_p(l, n, tp, &r1) != reference(l, n, tp, &r2)) same = 0;
    }
    CHECK(same, "400 random draws match the reference implementation");

    /* non-finite logits: +inf makes exp(logit - max) NaN; the result must be the argmax, not UB */
    {
        float bad[6]          = {0.5f, 1.0f, INFINITY, -2.0f, 0.1f, INFINITY};
        unsigned long long rs = 5;
        int pick              = sample_top_p(bad, 6, 0.9f, &rs);
        CHECK(pick == 2, "a +inf logit falls back to argmax (first maximum)");
        float nanv[4] = {NAN, 1.0f, 2.0f, 3.0f};
        rs            = 6;
        pick          = sample_top_p(nanv, 4, 0.9f, &rs);
        CHECK(pick >= 0 && pick < 4, "a NaN logit returns a valid index");
        float ninf[3] = {-INFINITY, -INFINITY, -INFINITY};
        rs            = 7;
        pick          = sample_top_p(ninf, 3, 0.9f, &rs);
        CHECK(pick >= 0 && pick < 3, "all -inf logits return a valid index");
    }

    fill(l, 60000, &s, 0.001f); /* flat: all 60000 are candidates */
    unsigned long long r1 = 5, r2 = 5;
    int a = sample_top_p(l, 60000, 0.95f, &r1), b = reference(l, 60000, 0.95f, &r2);
    CHECK(a == b && a >= 0 && a < 60000,
          "a flat 60000-token distribution (beyond the stack buffer) matches the reference");

    free(l);
    int big               = 250000; /* above the old 200000 limit */
    l                     = (float *)calloc((size_t)big, sizeof(float));
    l[123456]             = 50.0f;
    unsigned long long r3 = 7;
    CHECK(sample_top_p(l, big, 0.9f, &r3) == 123456,
          "a vocabulary above 200000 is sampled, not silently argmax'd");
    free(l);

    float one[1]          = {0.0f};
    unsigned long long r4 = 3;
    CHECK(sample_top_p(one, 1, 0.9f, &r4) == 0 && sample_top_p(one, 0, 0.9f, &r4) == 0,
          "tiny and empty vocabularies are safe");

    pthread_t th[4];
    Job jobs[4];
    for (int t = 0; t < 4; t++) {
        jobs[t].seed = 1000ULL * (unsigned)(t + 1);
        pthread_create(&th[t], NULL, worker, &jobs[t]);
    }
    int all = 1;
    for (int t = 0; t < 4; t++) {
        pthread_join(th[t], NULL);
        all &= jobs[t].ok;
    }
    CHECK(all, "4 threads sampling at once each match the reference (no shared buffer)");
    printf(g_fail ? "top_p: %d FAILED\n" : "top_p: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
