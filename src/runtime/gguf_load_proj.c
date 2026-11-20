/* Projection loader of the dense GGUF path: picks, per tensor, the zero-copy layout the matmul
 * kernels read (F16 / F32 / Q4_K / Q6_K), repacks BitNet I2_S and wide Q4_K, or dequantises. */

#include <stdio.h>
#include <stdlib.h>

#include "core/gguf_i2s.h"
#include "core/log.h"
#include "core/unpack.h"
#include "math/matmul_q4k_x8.h"
#include "memory/mapped_file.h"
#include "runtime/gguf_load_util.h"

/* Zero-copy: the tensor's bytes live in the mmap for the model's lifetime. */
static void use_mapped(tn_i8 **field, int *type_out, int l, const GGUFTensor *t, int weight_type) {
    field[l]    = (tn_i8 *)t->data;
    type_out[l] = weight_type;
}

/* BitNet I2_S: lossless repack into the engine's packed ternary rows (2026-10-18), so it runs on
 * the native ternary kernels. The raw tensor is unused afterwards and released. */
static TernaryError repack_i2s(const GGUFTensor *t, const char *name, int d_rows, int n_inner,
                               GGUFWeightStore *store, tn_i8 **field, int *type_out,
                               float *scale_out, int l) {
    size_t n_elems = (size_t)d_rows * (size_t)n_inner;
    size_t pb      = (size_t)d_rows * packed_bytes(n_inner);
    if (t->size_bytes < tn_i2s_tensor_bytes(n_elems)) {
        TN_LOG_ERR("gguf_loader", "[gguf_loader] I2_S tensor '%s' too small\n", name);
        return TN_ERR_INVALID_WEIGHTS;
    }
    uint8_t *pk = (uint8_t *)malloc(pb);
    if (!pk) return TN_ERR_OOM;
    if (tn_i2s_repack(pk, &scale_out[l], (const uint8_t *)t->data, d_rows, n_inner) != 0) {
        free(pk);
        TN_LOG_ERR("gguf_loader",
                   "[gguf_loader] I2_S tensor '%s' malformed (needs columns %% 128 == 0)\n", name);
        return TN_ERR_INVALID_WEIGHTS;
    }
    if (gguf_ld_store_add(store, pk) != 0) {
        free(pk);
        return TN_ERR_OOM;
    }
    mapped_file_release(t->data, t->size_bytes);
    field[l]    = (tn_i8 *)pk;
    type_out[l] = WEIGHT_TYPE_TERNARY;
    return TN_OK;
}

/* Q4_K with d_rows a multiple of 8: repacked once, at load time (not per token), into the
 * x8-interleaved layout of the real multi-row SIMD-lane GEMV (matmul_q4k_x8.c), matching
 * llama.cpp's actual technique rather than an approximation of it. */
static TernaryError repack_q4k_x8(const GGUFTensor *t, int d_rows, int n_inner,
                                  GGUFWeightStore *store, tn_i8 **field, int *type_out, int l) {
    int n_blocks     = (int)((size_t)n_inner / 256);
    int groups       = (int)((size_t)d_rows / 8);
    size_t row_bytes = (size_t)n_blocks * 144;
    TnQ4KX8Block *xbuf =
        (TnQ4KX8Block *)malloc((size_t)groups * (size_t)n_blocks * sizeof(TnQ4KX8Block));
    if (!xbuf) return TN_ERR_OOM;
    for (int g = 0; g < groups; g++) {
        const uint8_t *rows[8];
        for (int i = 0; i < 8; i++)
            rows[i] = (const uint8_t *)t->data + (size_t)(g * 8 + i) * row_bytes;
        tn_q4k_repack_x8(xbuf + (size_t)g * n_blocks, rows, n_blocks);
    }
    if (gguf_ld_store_add(store, xbuf) != 0) {
        free(xbuf);
        return TN_ERR_OOM;
    }
    field[l]    = (tn_i8 *)xbuf;
    type_out[l] = WEIGHT_TYPE_Q4K_X8;
    return TN_OK;
}

/* Loads "blk.<l>.<tname>" (d_rows output rows, n_inner input columns) into field[l] and records
 * its layout in type_out[l] (per layer and per PROJECTION, not a model-wide flag: Q4_K_M-style
 * files give attn_v / ffn_down a different quant type than the rest, and that varies by layer;
 * see weights.h). Q4_K is zero-copy (2026-07-30: dequantising every Q4_K tensor to F32 needed
 * ~28 GB heap for an 8B model and was OOM-killed); other quantised types are dequantised to a
 * heap F32 buffer. d_rows and n_inner are separate because the x8 repack needs both. */
TernaryError gguf_ld_dense_projection(const GGUFHeader *hdr, GGUFWeightStore *store, int l,
                                      const char *tname, int d_rows, int n_inner, tn_i8 **field,
                                      int *type_out, float *scale_out) {
    char name_buf[128];
    snprintf(name_buf, sizeof(name_buf), "blk.%d.%s", l, tname);
    const GGUFTensor *t = gguf_find_tensor(hdr, name_buf);
    if (!t) {
        TN_LOG_ERR("gguf_loader", "[gguf_loader] missing tensor '%s'\n", name_buf);
        return TN_ERR_INVALID_WEIGHTS;
    }
    size_t n_elems = (size_t)d_rows * (size_t)n_inner;
    if (t->type == GGUF_TYPE_I2_S)
        return repack_i2s(t, name_buf, d_rows, n_inner, store, field, type_out, scale_out, l);
    if (t->type == GGUF_TYPE_F16) {
        use_mapped(field, type_out, l, t, WEIGHT_TYPE_F16);
    } else if (t->type == GGUF_TYPE_F32) {
        use_mapped(field, type_out, l, t, WEIGHT_TYPE_F32);
    } else if (t->type == GGUF_TYPE_Q4_K && n_elems % 256 == 0 && (size_t)n_inner % 256 == 0 &&
               (size_t)d_rows % 8 == 0) {
        return repack_q4k_x8(t, d_rows, n_inner, store, field, type_out, l);
    } else if (t->type == GGUF_TYPE_Q4_K && n_elems % 256 == 0) {
        use_mapped(field, type_out, l, t, WEIGHT_TYPE_Q4K);
    } else if (t->type == GGUF_TYPE_Q6_K && (size_t)n_inner % 256 == 0) {
        use_mapped(field, type_out, l, t, WEIGHT_TYPE_Q6K); /* fused kernel, matmul_q6k.c */
    } else {
        float *f = gguf_ld_tensor_to_f32(t, n_elems, store);
        if (!f) return TN_ERR_INVALID_WEIGHTS;
        field[l]    = (tn_i8 *)f;
        type_out[l] = WEIGHT_TYPE_F32; /* dequantised to F32 on the heap */
    }
    return TN_OK;
}
