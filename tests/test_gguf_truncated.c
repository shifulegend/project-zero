/*
 * tests/test_gguf_truncated.c
 *
 * A GGUF whose tensor data does not fit in the file (truncated download) used to crash the loader:
 * the reader leaves such a tensor's data NULL and the dequantiser read through it (SEGV in
 * gguf_dequant_q8_0 on a 5 MB prefix of SmolLM2). weights_from_gguf() must refuse it with an error
 * and leave nothing for the caller to free. Hand-builds an in-memory header, no file.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/config.h"
#include "core/gguf_reader.h"
#include "core/weights.h"
#include "runtime/gguf_loader.h"

static int g_fail;
#define CHECK(c, msg)                                                                              \
    do {                                                                                           \
        if (!(c)) {                                                                                \
            printf("FAIL: %s\n", msg);                                                             \
            g_fail++;                                                                              \
        } else                                                                                     \
            printf("ok: %s\n", msg);                                                               \
    } while (0)

static GGUFHeader *make_header(const char *arch, int n_present, int truncate_at) {
    GGUFHeader *h = (GGUFHeader *)calloc(1, sizeof(*h));
    snprintf(h->arch, sizeof(h->arch), "%s", arch);
    static const uint8_t bytes[64] = {0};
    h->n_tensors                   = (uint64_t)n_present;
    for (int i = 0; i < n_present; i++) {
        snprintf(h->tensors[i].name, sizeof(h->tensors[i].name), "tensor.%d", i);
        h->tensors[i].type       = GGUF_TYPE_F32;
        h->tensors[i].data       = i == truncate_at ? NULL : bytes;
        h->tensors[i].size_bytes = i == truncate_at ? 0 : sizeof(bytes);
    }
    return h;
}

int main(void) {
    static const char *const archs[] = {"llama", "qwen3moe", "deepseek2", "mamba", "qwen35"};
    Config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.dim        = 64;
    cfg.hidden_dim = 128;
    cfg.n_layers   = 1;
    cfg.n_heads    = 4;
    cfg.n_kv_heads = 4;
    cfg.vocab_size = 32;
    cfg.seq_len    = 16;
    for (size_t a = 0; a < sizeof(archs) / sizeof(archs[0]); a++) {
        for (int cut = 0; cut < 3; cut++) {
            GGUFHeader *h = make_header(archs[a], 3, cut);
            TransformerWeights w;
            memset(&w, 0, sizeof(w));
            GGUFWeightStore *store = NULL;
            TernaryError e         = weights_from_gguf(&w, &cfg, h, &store);
            char msg[96];
            snprintf(msg, sizeof(msg), "%s: tensor %d missing -> refused, no crash", archs[a], cut);
            CHECK(e == TN_ERR_INVALID_WEIGHTS && store == NULL, msg);
            free(h);
        }
    }
    printf("=== %s ===\n", g_fail ? "FAILED" : "gguf truncated tests passed");
    return g_fail ? 1 : 0;
}
