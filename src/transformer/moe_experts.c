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
#include "math/matmul_q6k.h"
#include "math/q8_0_act.h"
#include "math/simd_dispatch.h"
#include "core/gguf_reader.h"
#include "core/platform.h"

#include <stdlib.h>
#include <string.h>

/* ── module scratch (moe_ffn_forward is not re-entrant) ─────────────────── */
static float          *s_g = NULL, *s_u = NULL;  /* [sum hdim] gate / up    */
static size_t          s_g_cap = 0, s_u_cap = 0;
static unsigned char  *s_hq = NULL;              /* quantized hidden vectors */
static size_t          s_hq_cap = 0;
static TnQ8KActBlock  *s_xk = NULL;              /* x as Q8K                 */
static TnQ8_0ActBlock *s_x0 = NULL;              /* x as Q8_0                */
static int             s_x_cap = 0;              /* elements of x covered    */
static int            *s_off = NULL;             /* [n_jobs+1] row offsets   */
static size_t         *s_hq_off = NULL;          /* [n_jobs] byte offsets    */
static int             s_jobs_cap = 0;

void moe_experts_free(void) {
    free(s_g); free(s_u); free(s_hq); free(s_xk); free(s_x0);
    free(s_off); free(s_hq_off);
    s_g = s_u = NULL; s_hq = NULL; s_xk = NULL; s_x0 = NULL;
    s_off = NULL; s_hq_off = NULL;
    s_g_cap = s_u_cap = s_hq_cap = 0; s_x_cap = 0; s_jobs_cap = 0;
}

static int is_kquant(int type) {
    return type == GGUF_TYPE_Q4_K || type == GGUF_TYPE_Q6_K;
}

static size_t row_bytes(int type, int n_inner) {
    switch (type) {
    case GGUF_TYPE_Q4_K: return (size_t)(n_inner / 256) * 144;
    case GGUF_TYPE_Q6_K: return (size_t)(n_inner / 256) * TN_Q6K_BYTES;
    case GGUF_TYPE_Q5_0: return (size_t)(n_inner / 32) * 22;
    case GGUF_TYPE_Q5_1: return (size_t)(n_inner / 32) * 24;
    case GGUF_TYPE_Q8_0: return (size_t)(n_inner / 32) * 34;
    default:             return 0;
    }
}

/* bytes of activation blocks needed to quantize an n-vector for `type` */
static size_t act_bytes(int type, int n) {
    return is_kquant(type) ? (size_t)(n / 256) * sizeof(TnQ8KActBlock)
                           : (size_t)(n / 32)  * sizeof(TnQ8_0ActBlock);
}

int moe_experts_type_ok(int type, int n_inner) {
    if (n_inner <= 0) return 0;
    if (is_kquant(type)) return n_inner % 256 == 0;
    if (type == GGUF_TYPE_Q5_0 || type == GGUF_TYPE_Q5_1 || type == GGUF_TYPE_Q8_0)
        return n_inner % 32 == 0;
    return 0;
}

int moe_experts_job_ok(const TnMoEExpertJob *j, int dim) {
    return j && j->w_gate && j->w_up && j->w_down && j->hdim > 0 &&
           moe_experts_type_ok(j->gate_type, dim) &&
           moe_experts_type_ok(j->up_type, dim) &&
           moe_experts_type_ok(j->down_type, j->hdim);
}

static inline float rowdot(int type, const uint8_t *row, const void *acts,
                           int n_inner, int vq4, int vq8) {
    switch (type) {
    case GGUF_TYPE_Q4_K: return tn_dot_q4k_row_q8k(row, (const TnQ8KActBlock *)acts, n_inner / 256, vq4);
    case GGUF_TYPE_Q6_K: return tn_dot_q6k_row_q8k(row, (const TnQ8KActBlock *)acts, n_inner / 256);
    case GGUF_TYPE_Q5_0: return tn_dot_q5_0_q8_0(row, (const TnQ8_0ActBlock *)acts, n_inner / 32, vq8);
    case GGUF_TYPE_Q5_1: return tn_dot_q5_1_q8_0(row, (const TnQ8_0ActBlock *)acts, n_inner / 32, vq8);
    case GGUF_TYPE_Q8_0: return tn_dot_q8_0_q8_0(row, (const TnQ8_0ActBlock *)acts, n_inner / 32, vq8);
    default:             return 0.0f;   /* unreachable: filtered by job_ok */
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

static inline const void *x_acts_for(int type) {
    return is_kquant(type) ? (const void *)s_xk : (const void *)s_x0;
}

/* ── phase 1: gate + up dots ─────────────────────────────────────────────── */
typedef struct {
    const TnMoEExpertJob *jobs;
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
    int j = 0;
    while (j + 1 < a->n_jobs && s_off[j + 1] <= start) j++;
    int r = start;
    while (r < end && j < a->n_jobs) {
        const TnMoEExpertJob *jb = a->jobs + j;
        const int base = s_off[j];
        const int jend = end < s_off[j + 1] ? end : s_off[j + 1];
        const size_t gb = row_bytes(jb->gate_type, a->dim);
        const size_t ub = row_bytes(jb->up_type, a->dim);
        const void *xg = x_acts_for(jb->gate_type);
        const void *xu = x_acts_for(jb->up_type);
        for (int rr = r; rr < jend; rr++) {
            const size_t li = (size_t)(rr - base);
            if (rr + MOE_PF_ROWS < jend) {
                prefetch_row(jb->w_gate + (li + MOE_PF_ROWS) * gb, gb);
                prefetch_row(jb->w_up   + (li + MOE_PF_ROWS) * ub, ub);
            }
            s_g[rr] = rowdot(jb->gate_type, jb->w_gate + li * gb, xg, a->dim, a->vq4, a->vq8);
            s_u[rr] = rowdot(jb->up_type,   jb->w_up   + li * ub, xu, a->dim, a->vq4, a->vq8);
        }
        r = jend;
        j++;
    }
}

/* ── phase 2: down projections summed into out ───────────────────────────── */
typedef struct {
    float *out;
    const TnMoEExpertJob *jobs;
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
        const size_t rb = row_bytes(jb->down_type, jb->hdim);
        const void *hq = s_hq + s_hq_off[j];
        const float wgt = jb->weight;
        for (int r = start; r < end; r++) {
            if (r + MOE_PF_ROWS < end)
                prefetch_row(jb->w_down + (size_t)(r + MOE_PF_ROWS) * rb, rb);
            a->out[r] += wgt * rowdot(jb->down_type, jb->w_down + (size_t)r * rb, hq,
                                      jb->hdim, a->vq4, a->vq8);
        }
    }
}

/* grow-only reallocation; returns 0 on success */
static int grow(void **p, size_t *cap, size_t need) {
    if (*cap >= need) return 0;
    void *n = malloc(need);
    if (!n) return -1;
    free(*p);
    *p = n;
    *cap = need;
    return 0;
}

int moe_experts_forward(float *out, const float *x,
                        const TnMoEExpertJob *jobs, int n_jobs,
                        int dim, int act_type, ThreadPool *tp) {
    if (!out || !x || !jobs || n_jobs <= 0 || dim <= 0) return -1;

    /* offsets */
    if (s_jobs_cap < n_jobs) {
        int *no = (int *)malloc((size_t)(n_jobs + 1) * sizeof(int));
        size_t *nh = (size_t *)malloc((size_t)n_jobs * sizeof(size_t));
        if (!no || !nh) { free(no); free(nh); return -1; }
        free(s_off); free(s_hq_off);
        s_off = no; s_hq_off = nh; s_jobs_cap = n_jobs;
    }
    int need_k = 0, need_0 = 0;
    size_t hq_total = 0;
    s_off[0] = 0;
    for (int j = 0; j < n_jobs; j++) {
        s_off[j + 1] = s_off[j] + jobs[j].hdim;
        s_hq_off[j]  = hq_total;
        hq_total    += act_bytes(jobs[j].down_type, jobs[j].hdim);
        need_k |= (is_kquant(jobs[j].gate_type) || is_kquant(jobs[j].up_type));
        need_0 |= (!is_kquant(jobs[j].gate_type) || !is_kquant(jobs[j].up_type));
    }
    const size_t rows = (size_t)s_off[n_jobs];
    if (grow((void **)&s_g, &s_g_cap, rows * sizeof(float)) ||
        grow((void **)&s_u, &s_u_cap, rows * sizeof(float))) return -1;
    if (grow((void **)&s_hq, &s_hq_cap, hq_total)) return -1;
    if (s_x_cap < dim) {
        TnQ8KActBlock  *xk = (TnQ8KActBlock *)malloc((size_t)(dim / 256 + 1) * sizeof(TnQ8KActBlock));
        TnQ8_0ActBlock *x0 = (TnQ8_0ActBlock *)malloc((size_t)(dim / 32 + 1) * sizeof(TnQ8_0ActBlock));
        if (!xk || !x0) { free(xk); free(x0); return -1; }
        free(s_xk); free(s_x0);
        s_xk = xk; s_x0 = x0; s_x_cap = dim;
    }

    /* quantize x once per needed format */
    if (need_k) tn_quantize_q8k(s_xk, x, dim / 256);
    if (need_0) tn_quantize_q8_0_act(s_x0, x, dim / 32);

    const int vq4 = tn_q4k_vnni_ok();
    const int vq8 = tn_q8_0_act_vnni_ok();

    MoEExpArgs a1 = { jobs, n_jobs, dim, vq4, vq8 };
    threadpool_dispatch_chunked(tp, phase1_task, &a1, (int)rows, 0);

    /* act(gate) * up, then quantize each hidden vector once for its down
     * type. The activation runs here over each WHOLE expert vector, not
     * inside phase 1's arbitrary row ranges: the SIMD silu/relu2 kernels
     * use a vector exp approximation with a scalar tail, so splitting a
     * vector at thread-dependent offsets made results depend on the thread
     * count (caught by tests/test_moe_experts.c). ~11K elements per layer. */
    for (int j = 0; j < n_jobs; j++) {
        void *dst = s_hq + s_hq_off[j];
        float *h = s_g + s_off[j];
        if (act_type == 1) tn_relu2(h, jobs[j].hdim);
        else               tn_silu(h, jobs[j].hdim);
        tn_vec_mul(h, h, s_u + s_off[j], jobs[j].hdim);
        if (is_kquant(jobs[j].down_type))
            tn_quantize_q8k((TnQ8KActBlock *)dst, h, jobs[j].hdim / 256);
        else
            tn_quantize_q8_0_act((TnQ8_0ActBlock *)dst, h, jobs[j].hdim / 32);
    }

    MoEDownArgs a2 = { out, jobs, n_jobs, dim, vq4, vq8 };
    threadpool_dispatch_chunked(tp, phase2_task, &a2, dim, 0);
    return 0;
}
