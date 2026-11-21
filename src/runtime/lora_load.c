#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/binary_io.h"
#include "core/fp16.h"
#include "core/log.h"
#include "core/lora.h"
#include "memory/mapped_file.h"

/* Reads `count` contiguous F16 values starting at byte offset *off, writes
 * them decoded to a freshly malloc'd F32 array, and advances *off. Returns
 * NULL (and leaves *off unchanged) on truncation or OOM. */
static float *read_f16_block(const void *base, size_t sz, size_t *off, size_t count) {
    size_t nbytes = count * sizeof(uint16_t);
    if (*off + nbytes > sz) return NULL;
    float *out = (float *)malloc(count * sizeof(float));
    if (!out) return NULL;
    const uint16_t *src = (const uint16_t *)((const char *)base + *off);
    for (size_t i = 0; i < count; i++) out[i] = tn_fp16_to_f32(src[i]);
    *off += nbytes;
    return out;
}

/* Target order must match include/core/lora.h's LoRaTargetBit enum and the
 * .lora.bin format doc's "in LoRaTargetBit order" note. */
typedef struct {
    LoRaTargetBit bit;
    LoRAModule **field; /* &lora->q, &lora->k, ... */
    int n;              /* A's row width (base model's input dim for this module) */
    int d;              /* B's row count (base model's output dim for this module) */
} LoraTargetSpec;

static void free_module_array(LoRAModule *arr, int n_layers) {
    if (!arr) return;
    for (int l = 0; l < n_layers; l++) {
        free((void *)arr[l].A);
        free((void *)arr[l].B);
    }
    free(arr);
}

TernaryError lora_load(LoRAWeights *lora, const char *path, int dim, int kv_dim, int hidden_dim,
                       int n_layers) {
    return lora_load_ex(lora, path, dim, dim, kv_dim, hidden_dim, n_layers);
}

/* The fields of a .lora.bin header (offsets in include/core/lora.h's format doc). */
typedef struct {
    int n_layers, rank;
    float alpha;
    int dim, hidden_dim, target_mask, kv_dim, q_dim;
} LoraFileHeader;

/* 1 when every sized field of the header is within its limit. */
static int lora_header_in_range(const LoraFileHeader *h) {
    const struct {
        int v, lo, hi;
    } limits[] = {{h->n_layers, 1, 512}, {h->rank, 1, 4096},         {h->dim, 1, 65536},
                  {h->kv_dim, 1, 65536}, {h->hidden_dim, 0, 262144}, {h->q_dim, 1, 262144}};
    for (size_t i = 0; i < sizeof(limits) / sizeof(limits[0]); i++)
        if (limits[i].v < limits[i].lo || limits[i].v > limits[i].hi) return 0;
    return 1;
}

/* Reads and range-checks the 64-byte header. 0 on success; every failure is logged. */
static int lora_read_header(const void *d, size_t sz, LoraFileHeader *h) {
    if (sz < 64) {
        TN_LOG_ERR("lora", "[lora] file too small\n");
        return -1;
    }
    uint32_t magic = tn_rd_u32(d, 0);
    if (magic != LORA_BIN_MAGIC) {
        TN_LOG_INF("lora", "[lora] bad magic 0x%08X (expected 0x%08X)\n", magic, LORA_BIN_MAGIC);
        return -1;
    }
    h->n_layers    = tn_rd_i32(d, 8);
    h->rank        = tn_rd_i32(d, 12);
    h->alpha       = tn_rd_f32(d, 16);
    h->dim         = tn_rd_i32(d, 20);
    h->hidden_dim  = tn_rd_i32(d, 24);
    h->target_mask = tn_rd_i32(d, 28);
    h->kv_dim      = tn_rd_i32(d, 32);
    h->q_dim       = tn_rd_i32(d, 36);
    if (h->q_dim == 0) h->q_dim = h->dim; /* pre-2026-10-19 files */

    if (!lora_header_in_range(h)) {
        TN_LOG_ERR("lora",
                   "[lora] invalid header (n_layers=%d rank=%d dim=%d kv_dim=%d hidden_dim=%d)\n",
                   h->n_layers, h->rank, h->dim, h->kv_dim, h->hidden_dim);
        return -1;
    }
    return 0;
}

/* 0 when the adapter was built for this base model; otherwise logs the difference. */
static int lora_matches_model(const LoraFileHeader *h, const char *path, int dim, int q_dim,
                              int kv_dim, int hidden_dim, int n_layers) {
    int same = h->n_layers == n_layers && h->dim == dim && h->hidden_dim == hidden_dim &&
               h->kv_dim == kv_dim && h->q_dim == q_dim;
    if (same) return 0;
    TN_LOG_INF("lora",
               "[lora] '%s' was built for a different base model "
               "(file: n_layers=%d dim=%d q_dim=%d kv_dim=%d hidden_dim=%d; "
               "model: n_layers=%d dim=%d q_dim=%d kv_dim=%d hidden_dim=%d)\n",
               path, h->n_layers, h->dim, h->q_dim, h->kv_dim, h->hidden_dim, n_layers, dim, q_dim,
               kv_dim, hidden_dim);
    return -1;
}

/* Reads one layer's A and B for a target into *m. 0 on success; frees what it read on failure. */
static int lora_read_layer(const void *d, size_t sz, size_t *off, const LoraTargetSpec *spec,
                           int rank, float scale, int t, int l, LoRAModule *m) {
    float *A = read_f16_block(d, sz, off, (size_t)rank * spec->n);
    if (!A) {
        TN_LOG_ERR("lora", "[lora] truncated/OOM reading target %d layer %d A\n", t, l);
        return -1;
    }
    float *B = read_f16_block(d, sz, off, (size_t)spec->d * rank);
    if (!B) {
        free(A);
        TN_LOG_ERR("lora", "[lora] truncated/OOM reading target %d layer %d B\n", t, l);
        return -1;
    }
    m->A     = A;
    m->B     = B;
    m->rank  = rank;
    m->scale = scale;
    return 0;
}

/* The module arrays of every target in the mask, in LoRaTargetBit order. 0 on success. */
static int lora_read_targets(LoRAWeights *lora, const void *d, size_t sz, const LoraFileHeader *h,
                             int q_dim, int kv_dim, int hidden_dim) {
    int dim     = h->dim;
    float scale = h->alpha / (float)h->rank;

    /* K/V project dim -> kv_dim, not dim -> dim, on a GQA model (kv_dim == dim on a plain MHA
     * model, so this is a no-op there) -- found via a real downloaded SmolLM2 LoRA adapter
     * corrupting this loader's reads on this project's own demo model (n_kv_heads=3 < n_heads=9),
     * see docs/ai/mistakes.md, 2026-09-24 entry. */
    const LoraTargetSpec specs[LORA_TARGET_COUNT] = {
        {LORA_TARGET_Q, &lora->q, dim, q_dim},
        {LORA_TARGET_K, &lora->k, dim, kv_dim},
        {LORA_TARGET_V, &lora->v, dim, kv_dim},
        {LORA_TARGET_O, &lora->o, q_dim, dim},
        {LORA_TARGET_GATE, &lora->gate, dim, hidden_dim},
        {LORA_TARGET_UP, &lora->up, dim, hidden_dim},
        {LORA_TARGET_DOWN, &lora->down, hidden_dim, dim},
    };
    size_t off = 64;
    for (int t = 0; t < LORA_TARGET_COUNT; t++) {
        if (!(h->target_mask & (1 << specs[t].bit))) continue;
        LoRAModule *arr = (LoRAModule *)calloc((size_t)h->n_layers, sizeof(LoRAModule));
        if (!arr) {
            TN_LOG_ERR("lora", "[lora] OOM allocating module array\n");
            return -1;
        }
        *specs[t].field = arr;
        for (int l = 0; l < h->n_layers; l++)
            if (lora_read_layer(d, sz, &off, &specs[t], h->rank, scale, t, l, &arr[l]) != 0)
                return -1;
    }
    return 0;
}

static void lora_free_arrays(LoRAWeights *lora, int n_layers) {
    free_module_array(lora->q, n_layers);
    free_module_array(lora->k, n_layers);
    free_module_array(lora->v, n_layers);
    free_module_array(lora->o, n_layers);
    free_module_array(lora->gate, n_layers);
    free_module_array(lora->up, n_layers);
    free_module_array(lora->down, n_layers);
    memset(lora, 0, sizeof(*lora));
}

TernaryError lora_load_ex(LoRAWeights *lora, const char *path, int dim, int q_dim, int kv_dim,
                          int hidden_dim, int n_layers) {
    if (!lora || !path) return TN_ERR_INVALID_ARGS;
    if (q_dim <= 0) q_dim = dim;
    memset(lora, 0, sizeof(*lora));

    MappedFile mf;
    TernaryError err = mapped_file_open(&mf, path);
    if (err != TN_OK) {
        TN_LOG_ERR("lora", "[lora] cannot open %s\n", path);
        return err;
    }

    LoraFileHeader h;
    if (lora_read_header(mf.data, mf.size, &h) != 0 ||
        lora_matches_model(&h, path, dim, q_dim, kv_dim, hidden_dim, n_layers) != 0) {
        mapped_file_close(&mf);
        return TN_ERR_INVALID_ARGS;
    }

    lora->n_layers   = n_layers;
    lora->rank       = h.rank;
    lora->alpha      = h.alpha;
    lora->dim        = dim;
    lora->hidden_dim = hidden_dim;

    if (lora_read_targets(lora, mf.data, mf.size, &h, q_dim, kv_dim, hidden_dim) != 0) {
        lora_free_arrays(lora, n_layers);
        mapped_file_close(&mf);
        return TN_ERR_INVALID_ARGS;
    }
    mapped_file_close(&mf);
    return TN_OK;
}

void lora_free(LoRAWeights *lora) {
    if (!lora) return;
    free_module_array(lora->q, lora->n_layers);
    free_module_array(lora->k, lora->n_layers);
    free_module_array(lora->v, lora->n_layers);
    free_module_array(lora->o, lora->n_layers);
    free_module_array(lora->gate, lora->n_layers);
    free_module_array(lora->up, lora->n_layers);
    free_module_array(lora->down, lora->n_layers);
    memset(lora, 0, sizeof(*lora));
}
