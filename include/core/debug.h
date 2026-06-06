#ifndef TN_DEBUG_H
#define TN_DEBUG_H

#include <math.h>
#include <stdio.h>

/*
 * Global verbose-debug flag.
 * Set to 1 by main.c when --verbose is passed.
 */
extern int   g_tn_verbose;

/*
 * Global tensor-dump file pointer.
 * When non-NULL, DBG_DUMP() writes CSV rows for comparison with
 * llama.cpp's llama_dump output. Set by main.c when --dump-tensors
 * is passed.  Format: layer,step,n_elem,v0..v7,mean,absmax
 */
extern FILE *g_dump_fp;

/* Print per-layer activation stats: max absolute value, NaN/Inf count */
static inline void dbg_vec_stats(const char *tag, const float *v, int n) {
    if (!g_tn_verbose) return;
    float mx = 0.0f; int bad = 0;
    for (int i = 0; i < n; i++) {
        float a = v[i] < 0 ? -v[i] : v[i];
        if (a > mx) mx = a;
        if (isnan(v[i]) || isinf(v[i])) bad++;
    }
    fprintf(stderr, "[DBG] %-30s  max_abs=%.4f  bad=%d\n", tag, mx, bad);
}

/*
 * Dump a float vector to the CSV dump file.
 * layer   : transformer layer index (-1 for non-layer steps)
 * step    : checkpoint name (e.g. "attn_norm", "ffn_moe_probs")
 * v       : float array
 * n       : number of elements
 */
static inline void dbg_dump_vec(int layer, const char *step,
                                 const float *v, int n) {
    if (!g_dump_fp || n <= 0) return;
    float mean = 0.0f, absmax = 0.0f;
    for (int i = 0; i < n; i++) {
        mean += v[i];
        float a = v[i] < 0.0f ? -v[i] : v[i];
        if (a > absmax) absmax = a;
    }
    mean /= (float)n;
    int nprint = n < 8 ? n : 8;
    fprintf(g_dump_fp, "%d,%s,%d", layer, step, n);
    for (int i = 0; i < nprint; i++) fprintf(g_dump_fp, ",%.8g", v[i]);
    for (int i = nprint; i < 8;   i++) fprintf(g_dump_fp, ",0");
    fprintf(g_dump_fp, ",%.8g,%.8g\n", mean, absmax);
    fflush(g_dump_fp);
}

#define DBG_VEC(tag, v, n)  do { if (g_tn_verbose) dbg_vec_stats((tag),(v),(n)); } while(0)
#define DBG_MSG(...)        do { if (g_tn_verbose) fprintf(stderr, "[DBG] " __VA_ARGS__); } while(0)

/* Dump macro — zero cost when g_dump_fp is NULL */
#define DBG_DUMP(layer, step, v, n)  do { if (g_dump_fp) dbg_dump_vec((layer),(step),(v),(n)); } while(0)

#endif /* TN_DEBUG_H */
