#include "core/lora.h"
#include "memory/mapped_file.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Internal: read a uint32/int32 at byte offset in mapped data ───────────
 * Mirrors src/multimodal/vision_weights_load.c's own rd32/rd32s helpers --
 * same small-per-TU-copy convention as f16_to_f32_scalar (matmul_f16.c
 * etc.), not a shared header. */
static uint32_t rd32(const void *base, size_t off) {
    uint32_t v;
    memcpy(&v, (const char *)base + off, 4);
    return v;
}

static int32_t rd32s(const void *base, size_t off) {
    int32_t v;
    memcpy(&v, (const char *)base + off, 4);
    return v;
}

static float rdf32(const void *base, size_t off) {
    float v;
    memcpy(&v, (const char *)base + off, 4);
    return v;
}

/* Scalar F16->F32 helper (mirrors f16_to_f32_scalar in matmul_f16.c /
 * batched_matmul_f16.c / gguf_loader.c). */
static inline float f16_to_f32_scalar(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = (uint32_t)(h & 0x3ff) << 13;
    uint32_t bits;
    if      (exp == 0)  bits = sign | mant;
    else if (exp == 31) bits = sign | 0x7f800000u | mant;
    else                bits = sign | ((exp + 112) << 23) | mant;
    float f; memcpy(&f, &bits, 4); return f;
}

/* Reads `count` contiguous F16 values starting at byte offset *off, writes
 * them decoded to a freshly malloc'd F32 array, and advances *off. Returns
 * NULL (and leaves *off unchanged) on truncation or OOM. */
static float *read_f16_block(const void *base, size_t sz, size_t *off, size_t count) {
    size_t nbytes = count * sizeof(uint16_t);
    if (*off + nbytes > sz) return NULL;
    float *out = (float *)malloc(count * sizeof(float));
    if (!out) return NULL;
    const uint16_t *src = (const uint16_t *)((const char *)base + *off);
    for (size_t i = 0; i < count; i++) out[i] = f16_to_f32_scalar(src[i]);
    *off += nbytes;
    return out;
}

/* Target order must match include/core/lora.h's LoRaTargetBit enum and the
 * .lora.bin format doc's "in LoRaTargetBit order" note. */
typedef struct {
    LoRaTargetBit bit;
    LoRAModule  **field;   /* &lora->q, &lora->k, ... */
    int           n;       /* A's row width (base model's input dim for this module) */
    int           d;       /* B's row count (base model's output dim for this module) */
} LoraTargetSpec;

static void free_module_array(LoRAModule *arr, int n_layers) {
    if (!arr) return;
    for (int l = 0; l < n_layers; l++) {
        free((void *)arr[l].A);
        free((void *)arr[l].B);
    }
    free(arr);
}

TernaryError lora_load(LoRAWeights *lora, const char *path,
                        int dim, int kv_dim, int hidden_dim, int n_layers) {
    if (!lora || !path) return TN_ERR_INVALID_ARGS;
    memset(lora, 0, sizeof(*lora));

    MappedFile mf;
    TernaryError err = mapped_file_open(&mf, path);
    if (err != TN_OK) {
        fprintf(stderr, "[lora] cannot open %s\n", path);
        return err;
    }

    const void *d  = mf.data;
    size_t      sz = mf.size;

    if (sz < 64) {
        fprintf(stderr, "[lora] file too small\n");
        goto fail;
    }

    uint32_t magic = rd32(d, 0);
    if (magic != LORA_BIN_MAGIC) {
        fprintf(stderr, "[lora] bad magic 0x%08X (expected 0x%08X)\n",
                magic, LORA_BIN_MAGIC);
        goto fail;
    }

    int   file_n_layers = rd32s(d, 8);
    int   rank           = rd32s(d, 12);
    float alpha           = rdf32(d, 16);
    int   file_dim        = rd32s(d, 20);
    int   file_hidden_dim = rd32s(d, 24);
    int   target_mask     = rd32s(d, 28);
    int   file_kv_dim      = rd32s(d, 32);

    if (file_n_layers < 1 || file_n_layers > 512 || rank < 1 || rank > 4096 ||
        file_dim < 1 || file_dim > 65536 || file_hidden_dim < 0 || file_hidden_dim > 262144 ||
        file_kv_dim < 1 || file_kv_dim > 65536) {
        fprintf(stderr, "[lora] invalid header (n_layers=%d rank=%d dim=%d kv_dim=%d hidden_dim=%d)\n",
                file_n_layers, rank, file_dim, file_kv_dim, file_hidden_dim);
        goto fail;
    }
    if (file_n_layers != n_layers || file_dim != dim || file_hidden_dim != hidden_dim ||
        file_kv_dim != kv_dim) {
        fprintf(stderr, "[lora] '%s' was built for a different base model "
                "(file: n_layers=%d dim=%d kv_dim=%d hidden_dim=%d; "
                "model: n_layers=%d dim=%d kv_dim=%d hidden_dim=%d)\n",
                path, file_n_layers, file_dim, file_kv_dim, file_hidden_dim,
                n_layers, dim, kv_dim, hidden_dim);
        goto fail;
    }

    lora->n_layers   = n_layers;
    lora->rank       = rank;
    lora->alpha      = alpha;
    lora->dim        = dim;
    lora->hidden_dim = hidden_dim;

    float scale = alpha / (float)rank;

    /* K/V project dim -> kv_dim, not dim -> dim, on a GQA model (kv_dim ==
     * dim on a plain MHA model, so this is a no-op there) -- found via a
     * real downloaded SmolLM2 LoRA adapter corrupting this loader's reads
     * on this project's own demo model (n_kv_heads=3 < n_heads=9), see
     * docs/ai/mistakes.md, 2026-09-24 entry. */
    const LoraTargetSpec specs[LORA_TARGET_COUNT] = {
        { LORA_TARGET_Q,    &lora->q,    dim,        dim        },
        { LORA_TARGET_K,    &lora->k,    dim,        kv_dim     },
        { LORA_TARGET_V,    &lora->v,    dim,        kv_dim     },
        { LORA_TARGET_O,    &lora->o,    dim,        dim        },
        { LORA_TARGET_GATE, &lora->gate, dim,        hidden_dim },
        { LORA_TARGET_UP,   &lora->up,   dim,        hidden_dim },
        { LORA_TARGET_DOWN, &lora->down, hidden_dim, dim        },
    };

    size_t off = 64;

    for (int t = 0; t < LORA_TARGET_COUNT; t++) {
        if (!(target_mask & (1 << specs[t].bit))) continue;

        LoRAModule *arr = (LoRAModule *)calloc((size_t)n_layers, sizeof(LoRAModule));
        if (!arr) { fprintf(stderr, "[lora] OOM allocating module array\n"); goto fail_free; }
        *specs[t].field = arr;

        for (int l = 0; l < n_layers; l++) {
            float *A = read_f16_block(d, sz, &off, (size_t)rank * specs[t].n);
            if (!A) { fprintf(stderr, "[lora] truncated/OOM reading target %d layer %d A\n", t, l); goto fail_free; }
            float *B = read_f16_block(d, sz, &off, (size_t)specs[t].d * rank);
            if (!B) { free(A); fprintf(stderr, "[lora] truncated/OOM reading target %d layer %d B\n", t, l); goto fail_free; }
            arr[l].A     = A;
            arr[l].B     = B;
            arr[l].rank  = rank;
            arr[l].scale = scale;
        }
    }

    mapped_file_close(&mf);
    return TN_OK;

fail_free:
    free_module_array(lora->q,    n_layers);
    free_module_array(lora->k,    n_layers);
    free_module_array(lora->v,    n_layers);
    free_module_array(lora->o,    n_layers);
    free_module_array(lora->gate, n_layers);
    free_module_array(lora->up,   n_layers);
    free_module_array(lora->down, n_layers);
    memset(lora, 0, sizeof(*lora));
fail:
    mapped_file_close(&mf);
    return TN_ERR_INVALID_ARGS;
}

void lora_free(LoRAWeights *lora) {
    if (!lora) return;
    free_module_array(lora->q,    lora->n_layers);
    free_module_array(lora->k,    lora->n_layers);
    free_module_array(lora->v,    lora->n_layers);
    free_module_array(lora->o,    lora->n_layers);
    free_module_array(lora->gate, lora->n_layers);
    free_module_array(lora->up,   lora->n_layers);
    free_module_array(lora->down, lora->n_layers);
    memset(lora, 0, sizeof(*lora));
}
