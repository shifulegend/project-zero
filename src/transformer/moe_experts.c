/*
 * moe_experts.c — fused two-dispatch routed/shared expert FFN (2026-10-21).
 * Design and rationale: include/transformer/moe_experts.h.
 *
 * Profile that motivated it (DeepSeek-V2-Lite Q4_K_S, 4 threads, this host):
 * routed experts 2.64 ms + shared expert 0.72 ms per MoE layer = 87 ms of a
 * 135 ms token; the Q5_0 down projection alone ran at ~9 GB/s because its
 * kernel decoded weights to F32 and ran FMAs against an F32 activation.
 */

#include "transformer/moe_experts.h"

#include <stdlib.h>
#include <string.h>

#include "core/gguf_reader.h"
#include "core/platform.h"
#include "math/matmul_q6k.h"
#include "math/q8_0_act.h"
#include "math/simd_vec_ops.h"
#include "memory/tls_scratch.h"

/* ── per-thread scratch ─────────────────────────────────────────────────────
 * One block per calling thread (memory/tls_scratch.h), carved into the arrays below on every
 * call and handed to the pool workers through the task argument structs: the workers run on other
 * threads and must not look the scratch up themselves. (It used to be file-scope statics, which
 * two concurrent callers corrupted.) */
static TnTlsScratch s_scratch = TN_TLS_SCRATCH_INIT;

typedef struct {
    float *g, *u;       /* [sum hdim] gate / up                  */
    unsigned char *hq;  /* quantized hidden vectors              */
    TnQ8KActBlock *xk;  /* x as Q8K                              */
    TnQ8_0ActBlock *x0; /* x as Q8_0                             */
    int *off;           /* [n_jobs+1] row offsets                */
    size_t *hq_off;     /* [n_jobs] byte offsets into hq         */
} MoEScratch;

void moe_experts_free(void) {
    tn_tls_scratch_release(&s_scratch);
}

/* The expert weight formats with an integer dot, one row each: the activation block they pair with
 * (Q8K for the K-quants, Q8_0 otherwise) and the weight block size. A new format is a new row here
 * plus a case in rowdot(); every size and acceptance check below only looks it up. */
typedef struct {
    int type;
    int kquant;         /* pairs with Q8K activation blocks (256 wide) instead of Q8_0 (32 wide) */
    int block_elems;    /* weights per block: 256 or 32 */
    size_t block_bytes; /* bytes per block */
} ExpertFormat;

static const ExpertFormat k_expert_formats[] = {
    {GGUF_TYPE_Q4_K, 1, 256, 144}, {GGUF_TYPE_Q6_K, 1, 256, TN_Q6K_BYTES},
    {GGUF_TYPE_Q5_0, 0, 32, 22},   {GGUF_TYPE_Q5_1, 0, 32, 24},
    {GGUF_TYPE_Q8_0, 0, 32, 34},
};

static const ExpertFormat *expert_format(int type) {
    for (size_t i = 0; i < sizeof(k_expert_formats) / sizeof(k_expert_formats[0]); i++)
        if (k_expert_formats[i].type == type) return &k_expert_formats[i];
    return NULL;
}

static int is_kquant(int type) {
    const ExpertFormat *k = expert_format(type);
    return k && k->kquant;
}

static size_t row_bytes(int type, int n_inner) {
    const ExpertFormat *k = expert_format(type);
    return k ? (size_t)(n_inner / k->block_elems) * k->block_bytes : 0;
}

/* bytes of activation blocks needed to quantize an n-vector for `type` */
static size_t act_bytes(int type, int n) {
    return is_kquant(type) ? (size_t)(n / 256) * sizeof(TnQ8KActBlock)
                           : (size_t)(n / 32) * sizeof(TnQ8_0ActBlock);
}

int moe_experts_type_ok(int type, int n_inner) {
    const ExpertFormat *k = expert_format(type);
    return k && n_inner > 0 && n_inner % k->block_elems == 0;
}

int moe_experts_job_ok(const TnMoEExpertJob *j, int dim) {
    return j && j->w_gate && j->w_up && j->w_down && j->hdim > 0 &&
           moe_experts_type_ok(j->gate_type, dim) && moe_experts_type_ok(j->up_type, dim) &&
           moe_experts_type_ok(j->down_type, j->hdim);
}

/* The per-row dot stays a switch: an indirect call here measured about 2.5% slower on Q4_K
 * expert GEMVs (docs/ai/change-trace.md 2026-11-21), and this is the innermost loop. */
static inline float rowdot(int type, const uint8_t *row, const void *acts, int n_inner, int vq4,
                           int vq8) {
    switch (type) {
        case GGUF_TYPE_Q4_K:
            return tn_dot_q4k_row_q8k(row, (const TnQ8KActBlock *)acts, n_inner / 256, vq4);
        case GGUF_TYPE_Q6_K:
            return tn_dot_q6k_row_q8k(row, (const TnQ8KActBlock *)acts, n_inner / 256);
        case GGUF_TYPE_Q5_0:
            return tn_dot_q5_0_q8_0(row, (const TnQ8_0ActBlock *)acts, n_inner / 32, vq8);
        case GGUF_TYPE_Q5_1:
            return tn_dot_q5_1_q8_0(row, (const TnQ8_0ActBlock *)acts, n_inner / 32, vq8);
        case GGUF_TYPE_Q8_0:
            return tn_dot_q8_0_q8_0(row, (const TnQ8_0ActBlock *)acts, n_inner / 32, vq8);
        default:
            return 0.0f; /* unreachable: filtered by job_ok */
    }
}

/* Software prefetch distance in rows (2026-10-21). Single-core streaming on
 * the dev host went 7.3 -> 9.3 GB/s with a ~512 B-ahead software prefetch;
 * the expert GEMVs are latency-bound per core, so each task prefetches the
 * row it will need MOE_PF_ROWS rows from now (gate + up, or down). */
#define MOE_PF_ROWS 2

static inline void prefetch_row(const uint8_t *p, size_t bytes) {
    for (size_t o = 0; o < bytes; o += 64) TN_PREFETCH_T1(p + o);
}

static inline const void *x_acts_for(const MoEScratch *sc, int type) {
    return is_kquant(type) ? (const void *)sc->xk : (const void *)sc->x0;
}

/* ── phase 1: gate + up dots ─────────────────────────────────────────────── */
typedef struct {
    const TnMoEExpertJob *jobs;
    const MoEScratch *sc;
    int n_jobs, dim, vq4, vq8;
} MoEExpArgs;

/* Both phases run under threadpool_dispatch_chunked(): slices pull row
 * chunks from a shared counter instead of owning a static 1/N range, so a
 * thread delayed by the VM (steal time) or by a slower mix of rows no longer
 * stalls the dispatch. Each row is computed by exactly one thread with the
 * same arithmetic, so results do not depend on the schedule. */
static void phase1_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    const MoEExpArgs *a = (const MoEExpArgs *)arg;
    int j               = 0;
    while (j + 1 < a->n_jobs && a->sc->off[j + 1] <= start) j++;
    int r = start;
    while (r < end && j < a->n_jobs) {
        const TnMoEExpertJob *jb = a->jobs + j;
        const int base           = a->sc->off[j];
        const int jend           = end < a->sc->off[j + 1] ? end : a->sc->off[j + 1];
        const size_t gb          = row_bytes(jb->gate_type, a->dim);
        const size_t ub          = row_bytes(jb->up_type, a->dim);
        const void *xg           = x_acts_for(a->sc, jb->gate_type);
        const void *xu           = x_acts_for(a->sc, jb->up_type);
        for (int rr = r; rr < jend; rr++) {
            const size_t li = (size_t)(rr - base);
            if (rr + MOE_PF_ROWS < jend) {
                prefetch_row(jb->w_gate + (li + MOE_PF_ROWS) * gb, gb);
                prefetch_row(jb->w_up + (li + MOE_PF_ROWS) * ub, ub);
            }
            a->sc->g[rr] = rowdot(jb->gate_type, jb->w_gate + li * gb, xg, a->dim, a->vq4, a->vq8);
            a->sc->u[rr] = rowdot(jb->up_type, jb->w_up + li * ub, xu, a->dim, a->vq4, a->vq8);
        }
        r = jend;
        j++;
    }
}

/* ── phase 2: down projections summed into out ───────────────────────────── */
typedef struct {
    float *out;
    const TnMoEExpertJob *jobs;
    const MoEScratch *sc;
    int n_jobs, dim, vq4, vq8;
} MoEDownArgs;

static void phase2_task(void *arg, int thread_id, int start, int end) {
    (void)thread_id;
    const MoEDownArgs *a = (const MoEDownArgs *)arg;
    /* expert-outer: each thread streams one contiguous row range of one
     * down matrix at a time (hardware-prefetch friendly); the out rows are
     * owned exclusively by this thread, so += is race-free. */
    for (int j = 0; j < a->n_jobs; j++) {
        const TnMoEExpertJob *jb = a->jobs + j;
        const size_t rb          = row_bytes(jb->down_type, jb->hdim);
        const void *hq           = a->sc->hq + a->sc->hq_off[j];
        const float wgt          = jb->weight;
        for (int r = start; r < end; r++) {
            if (r + MOE_PF_ROWS < end)
                prefetch_row(jb->w_down + (size_t)(r + MOE_PF_ROWS) * rb, rb);
            a->out[r] += wgt * rowdot(jb->down_type, jb->w_down + (size_t)r * rb, hq, jb->hdim,
                                      a->vq4, a->vq8);
        }
    }
}

/* Rounds a segment size up so the next one starts 64-byte aligned. */
static size_t seg(size_t bytes) {
    return (bytes + 63) & ~(size_t)63;
}

int moe_experts_forward(float *out, const float *x, const TnMoEExpertJob *jobs, int n_jobs, int dim,
                        int act_type, ThreadPool *tp) {
    if (!out || !x || !jobs || n_jobs <= 0 || dim <= 0) return -1;

    /* sizes */
    int need_k = 0, need_0 = 0;
    size_t hq_total = 0, rows = 0;
    for (int j = 0; j < n_jobs; j++) {
        rows += (size_t)jobs[j].hdim;
        hq_total += act_bytes(jobs[j].down_type, jobs[j].hdim);
        need_k |= (is_kquant(jobs[j].gate_type) || is_kquant(jobs[j].up_type));
        need_0 |= (!is_kquant(jobs[j].gate_type) || !is_kquant(jobs[j].up_type));
    }
    const size_t xk_bytes = (size_t)(dim / 256 + 1) * sizeof(TnQ8KActBlock);
    const size_t x0_bytes = (size_t)(dim / 32 + 1) * sizeof(TnQ8_0ActBlock);
    const size_t total    = seg(rows * sizeof(float)) * 2 + seg(hq_total) + seg(xk_bytes) +
                         seg(x0_bytes) + seg((size_t)(n_jobs + 1) * sizeof(int)) +
                         seg((size_t)n_jobs * sizeof(size_t));
    unsigned char *blk = (unsigned char *)tn_tls_scratch(&s_scratch, total);
    if (!blk) return -1;

    MoEScratch sc;
    memset(&sc, 0, sizeof(sc));
    sc.g = (float *)blk;
    blk += seg(rows * sizeof(float));
    sc.u = (float *)blk;
    blk += seg(rows * sizeof(float));
    sc.hq = blk;
    blk += seg(hq_total);
    sc.xk = (TnQ8KActBlock *)blk;
    blk += seg(xk_bytes);
    sc.x0 = (TnQ8_0ActBlock *)blk;
    blk += seg(x0_bytes);
    sc.off = (int *)blk;
    blk += seg((size_t)(n_jobs + 1) * sizeof(int));
    sc.hq_off = (size_t *)blk;

    size_t hq_pos = 0;
    sc.off[0]     = 0;
    for (int j = 0; j < n_jobs; j++) {
        sc.off[j + 1] = sc.off[j] + jobs[j].hdim;
        sc.hq_off[j]  = hq_pos;
        hq_pos += act_bytes(jobs[j].down_type, jobs[j].hdim);
    }

    /* quantize x once per needed format */
    if (need_k) tn_quantize_q8k(sc.xk, x, dim / 256);
    if (need_0) tn_quantize_q8_0_act(sc.x0, x, dim / 32);

    const int vq4 = tn_q4k_vnni_ok();
    const int vq8 = tn_q8_0_act_vnni_ok();

    MoEExpArgs a1 = {jobs, &sc, n_jobs, dim, vq4, vq8};
    threadpool_dispatch_chunked(tp, phase1_task, &a1, (int)rows, 0);

    /* act(gate) * up, then quantize each hidden vector once for its down
     * type. The activation runs here over each WHOLE expert vector, not
     * inside phase 1's arbitrary row ranges: the SIMD silu/relu2 kernels
     * use a vector exp approximation with a scalar tail, so splitting a
     * vector at thread-dependent offsets made results depend on the thread
     * count (caught by tests/test_moe_experts.c). ~11K elements per layer. */
    for (int j = 0; j < n_jobs; j++) {
        void *dst = sc.hq + sc.hq_off[j];
        float *h  = sc.g + sc.off[j];
        if (act_type == 1)
            tn_relu2(h, jobs[j].hdim);
        else
            tn_silu(h, jobs[j].hdim);
        tn_vec_mul(h, h, sc.u + sc.off[j], jobs[j].hdim);
        if (is_kquant(jobs[j].down_type))
            tn_quantize_q8k((TnQ8KActBlock *)dst, h, jobs[j].hdim / 256);
        else
            tn_quantize_q8_0_act((TnQ8_0ActBlock *)dst, h, jobs[j].hdim / 32);
    }

    MoEDownArgs a2 = {out, jobs, &sc, n_jobs, dim, vq4, vq8};
    threadpool_dispatch_chunked(tp, phase2_task, &a2, dim, 0);
    return 0;
}
