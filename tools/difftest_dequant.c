/*
 * difftest_dequant.c — TS-1.1: differential test of our gguf_dequant_* functions
 * against ggml's own quantize/dequantize functions (linked from a real
 * llama.cpp/ggml build, not reimplemented).
 *
 * For each supported quant type: generate a deterministic pseudo-random
 * float32 row, quantize it with ggml_quantize_chunk(), dequantize the exact
 * same bytes with ggml's own dequantize function (ref) and with our
 * gguf_dequant function (ours), and assert bit-exact equality.
 *
 * Build: see tools/Makefile.difftest (links against the llama.cpp/ggml
 * shared libs built in /tmp/llama-ref/build).
 */
#include "core/gguf_quant.h"

#include "ggml.h"
#include "ggml-quants.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N_ELEMS 4096

static void fill_pseudo_random(float *x, int64_t n, uint32_t seed) {
    /* xorshift32 -- deterministic, no libc rand() state across platforms */
    uint32_t s = seed;
    for (int64_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        /* map to roughly [-4, 4), avoid all-zero pathological rows */
        x[i] = ((float)(s % 200000) / 200000.0f - 0.5f) * 8.0f;
    }
}

typedef void (*dequant_fn)(float *out, const void *data, size_t n_elems);

typedef struct {
    enum ggml_type ggml_type;
    const char *name;
    dequant_fn ours;
    bool needs_imatrix;
} Case;

static int g_pass = 0, g_fail = 0;

static void run_case(const Case *c, const float *src, const float *imatrix) {
    size_t row_size = ggml_row_size(c->ggml_type, N_ELEMS);
    void *qbuf = malloc(row_size);
    float *ref = malloc(N_ELEMS * sizeof(float));
    float *ours = malloc(N_ELEMS * sizeof(float));
    if (!qbuf || !ref || !ours) { fprintf(stderr, "alloc failure\n"); exit(1); }

    ggml_quantize_chunk(c->ggml_type, src, qbuf, 0, 1, N_ELEMS,
                         c->needs_imatrix ? imatrix : NULL);

    /* ggml's own reference dequant for this type */
    switch (c->ggml_type) {
        case GGML_TYPE_F32:     memcpy(ref, qbuf, N_ELEMS * sizeof(float)); break;
        case GGML_TYPE_F16:     ggml_fp16_to_fp32_row((const ggml_fp16_t *)qbuf, ref, N_ELEMS); break;
        case GGML_TYPE_BF16:    ggml_bf16_to_fp32_row((const ggml_bf16_t *)qbuf, ref, N_ELEMS); break;
        case GGML_TYPE_Q4_0:    dequantize_row_q4_0((const block_q4_0 *)qbuf, ref, N_ELEMS); break;
        case GGML_TYPE_Q4_1:    dequantize_row_q4_1((const block_q4_1 *)qbuf, ref, N_ELEMS); break;
        case GGML_TYPE_Q5_0:    dequantize_row_q5_0((const block_q5_0 *)qbuf, ref, N_ELEMS); break;
        case GGML_TYPE_Q5_1:    dequantize_row_q5_1((const block_q5_1 *)qbuf, ref, N_ELEMS); break;
        case GGML_TYPE_Q8_0:    dequantize_row_q8_0((const block_q8_0 *)qbuf, ref, N_ELEMS); break;
        case GGML_TYPE_Q2_K:    dequantize_row_q2_K((const block_q2_K *)qbuf, ref, N_ELEMS); break;
        case GGML_TYPE_Q3_K:    dequantize_row_q3_K((const block_q3_K *)qbuf, ref, N_ELEMS); break;
        case GGML_TYPE_Q4_K:    dequantize_row_q4_K((const block_q4_K *)qbuf, ref, N_ELEMS); break;
        case GGML_TYPE_Q5_K:    dequantize_row_q5_K((const block_q5_K *)qbuf, ref, N_ELEMS); break;
        case GGML_TYPE_Q6_K:    dequantize_row_q6_K((const block_q6_K *)qbuf, ref, N_ELEMS); break;
        case GGML_TYPE_IQ2_XXS: dequantize_row_iq2_xxs((const block_iq2_xxs *)qbuf, ref, N_ELEMS); break;
        case GGML_TYPE_IQ2_XS:  dequantize_row_iq2_xs((const block_iq2_xs *)qbuf, ref, N_ELEMS); break;
        case GGML_TYPE_IQ2_S:   dequantize_row_iq2_s((const block_iq2_s *)qbuf, ref, N_ELEMS); break;
        case GGML_TYPE_IQ3_XXS: dequantize_row_iq3_xxs((const block_iq3_xxs *)qbuf, ref, N_ELEMS); break;
        case GGML_TYPE_IQ3_S:   dequantize_row_iq3_s((const block_iq3_s *)qbuf, ref, N_ELEMS); break;
        case GGML_TYPE_IQ1_S:   dequantize_row_iq1_s((const block_iq1_s *)qbuf, ref, N_ELEMS); break;
        case GGML_TYPE_IQ1_M:   dequantize_row_iq1_m((const block_iq1_m *)qbuf, ref, N_ELEMS); break;
        case GGML_TYPE_IQ4_NL:  dequantize_row_iq4_nl((const block_iq4_nl *)qbuf, ref, N_ELEMS); break;
        case GGML_TYPE_IQ4_XS:  dequantize_row_iq4_xs((const block_iq4_xs *)qbuf, ref, N_ELEMS); break;
        default: fprintf(stderr, "no ref dequant wired for %s\n", c->name); exit(1);
    }

    c->ours(ours, qbuf, N_ELEMS);

    double max_abs_diff = 0.0;
    int64_t n_diff = 0;
    for (int64_t i = 0; i < N_ELEMS; i++) {
        double d = fabs((double)ref[i] - (double)ours[i]);
        if (d > max_abs_diff) max_abs_diff = d;
        if (ref[i] != ours[i]) n_diff++;
    }

    if (n_diff == 0) {
        printf("[PASS] %-10s bit-exact over %d elems\n", c->name, N_ELEMS);
        g_pass++;
    } else {
        printf("[FAIL] %-10s %ld/%d elems differ, max|diff|=%.9g\n",
               c->name, (long)n_diff, N_ELEMS, max_abs_diff);
        g_fail++;
    }

    free(qbuf); free(ref); free(ours);
}

int main(void) {
    float *src = malloc(N_ELEMS * sizeof(float));
    float *imatrix = malloc(N_ELEMS * sizeof(float));
    fill_pseudo_random(src, N_ELEMS, 0xC0FFEEu);
    for (int64_t i = 0; i < N_ELEMS; i++) imatrix[i] = 1.0f;

    Case cases[] = {
        { GGML_TYPE_F32,     "F32",     NULL,                     false },
        { GGML_TYPE_BF16,    "BF16",    gguf_dequant_bf16,         false },
        { GGML_TYPE_Q4_0,    "Q4_0",    gguf_dequant_q4_0,         false },
        { GGML_TYPE_Q4_1,    "Q4_1",    gguf_dequant_q4_1,         false },
        { GGML_TYPE_Q5_0,    "Q5_0",    gguf_dequant_q5_0,         false },
        { GGML_TYPE_Q5_1,    "Q5_1",    gguf_dequant_q5_1,         false },
        { GGML_TYPE_Q8_0,    "Q8_0",    gguf_dequant_q8_0,         false },
        { GGML_TYPE_Q2_K,    "Q2_K",    gguf_dequant_q2_k,         false },
        { GGML_TYPE_Q3_K,    "Q3_K",    gguf_dequant_q3_k,         false },
        { GGML_TYPE_Q4_K,    "Q4_K",    gguf_dequant_q4_k,         false },
        { GGML_TYPE_Q5_K,    "Q5_K",    gguf_dequant_q5_k,         false },
        { GGML_TYPE_Q6_K,    "Q6_K",    gguf_dequant_q6_k,         false },
        { GGML_TYPE_IQ2_XXS, "IQ2_XXS", gguf_dequant_iq2_xxs,      true  },
        { GGML_TYPE_IQ2_XS,  "IQ2_XS",  gguf_dequant_iq2_xs,       true  },
        { GGML_TYPE_IQ2_S,   "IQ2_S",   gguf_dequant_iq2_s,        false },
        { GGML_TYPE_IQ3_XXS, "IQ3_XXS", gguf_dequant_iq3_xxs,      false },
        { GGML_TYPE_IQ3_S,   "IQ3_S",   gguf_dequant_iq3_s,        false },
        { GGML_TYPE_IQ1_S,   "IQ1_S",   gguf_dequant_iq1_s,        true  },
        { GGML_TYPE_IQ1_M,   "IQ1_M",   gguf_dequant_iq1_m,        false },
        { GGML_TYPE_IQ4_NL,  "IQ4_NL",  gguf_dequant_iq4_nl,       false },
        { GGML_TYPE_IQ4_XS,  "IQ4_XS",  gguf_dequant_iq4_xs,       false },
    };

    /* F32 has no meaningful "ours" dequant (passthrough) -- skip via ours==NULL */
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        if (cases[i].ours == NULL) continue;
        run_case(&cases[i], src, imatrix);
    }

    printf("\n=== difftest_dequant: %d passed, %d failed (of %d types) ===\n",
           g_pass, g_fail, g_pass + g_fail);

    free(src); free(imatrix);
    return g_fail == 0 ? 0 : 1;
}
