/*
 * tests/test_gguf_loader_qwen3moe.c
 *
 * Fast, deterministic loader test for Qwen3-MoE (qwen3moe) GGUF support
 * (fixes issue #32: "missing tensor 'blk.0.ffn_gate.weight'"). Hand-builds
 * a tiny synthetic in-memory GGUFHeader (no real .gguf file, no mmap, no
 * multi-GB download) and exercises config_from_gguf/moe_config_from_gguf/
 * weights_from_gguf directly, then runs one real forward-pass token through
 * the new qwen3moe_attention_forward() path — all under make test's
 * ASan/UBSan build. Real-model golden-output verification against
 * Qwen3-30B-A3B-Q4_K_M.gguf is a separate, non-CI follow-up (see
 * docs/ai/decision-log.md) — GGUFHeader's tensors[]/meta[] are plain
 * fixed-size embedded arrays (see include/core/gguf_reader.h), so this
 * synthetic fixture needs no byte-level GGUF serialization at all.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <math.h>
#include "core/gguf_reader.h"
#include "core/gguf_loader.h"
#include "core/moe_config.h"
#include "core/moe_weights.h"
#include "core/weights.h"
#include "core/run_state.h"
#include "transformer/forward.h"
#include "math/simd_dispatch.h"
#include "threading/thread_pool.h"

#define PASS(name)  printf("[PASS] %s\n", name)
#define FAIL(name, msg) do { printf("[FAIL] %s: %s\n", name, msg); g_failures++; } while(0)

static int g_failures = 0;

/* Tiny synthetic model dims — deliberately small, just large enough to
 * exercise every code path (2 heads, 1 kv head -> GQA, 2 experts -> stride
 * slicing, head_dim=4 independent of dim/n_heads=4 by coincidence here, but
 * the loader/attention code never assumes they're equal). */
#define T_DIM        8
#define T_LAYERS     1
#define T_HEADS      2
#define T_KV_HEADS   1
#define T_HEAD_DIM   4
#define T_HIDDEN     8
#define T_VOCAB      6
#define T_EXPERTS    2
#define T_EXPERTS_PER_TOK 1
#define T_EXP_HID    4

#define T_Q_ROWS  (T_HEADS * T_HEAD_DIM)
#define T_KV_ROWS (T_KV_HEADS * T_HEAD_DIM)

static void add_meta_u32(GGUFHeader *hdr, const char *key, uint32_t val) {
    GGUFMeta *m = &hdr->meta[hdr->n_meta++];
    memset(m, 0, sizeof(*m));
    strncpy(m->key, key, GGUF_MAX_KEY_LEN - 1);
    m->val_type = GGUF_VAL_UINT32;
    m->val.u32 = val;
}

/* Allocates and fills an F32 tensor with `n_elems` floats, registers it in
 * hdr->tensors[], and returns the buffer (caller must free after the test —
 * GGUFHeader.tensors[].data is just a pointer, no ownership semantics of
 * its own, matching the real zero-copy-mmap convention this loader
 * otherwise relies on). Fill pattern: buf[i] = base + i * 0.01f, so
 * different tensors/experts produce distinguishable (non-aliased) values. */
static float *add_f32_tensor(GGUFHeader *hdr, const char *name,
                              uint64_t n_elems, float base,
                              uint64_t dim0, uint64_t dim1) {
    float *buf = (float *)malloc(n_elems * sizeof(float));
    for (uint64_t i = 0; i < n_elems; i++) buf[i] = base + (float)i * 0.01f;

    GGUFTensor *t = &hdr->tensors[hdr->n_tensors++];
    memset(t, 0, sizeof(*t));
    strncpy(t->name, name, GGUF_MAX_KEY_LEN - 1);
    t->type = GGUF_TYPE_F32;
    t->data = buf;
    t->size_bytes = n_elems * sizeof(float);
    if (dim1 > 0) {
        t->n_dims = 2;
        t->dims[0] = dim0;
        t->dims[1] = dim1;
    } else {
        t->n_dims = 1;
        t->dims[0] = dim0;
    }
    return buf;
}

/* Tracks every heap buffer add_f32_tensor() allocated, so the test can free
 * them all at the end regardless of which assertion path returns early. */
#define MAX_BUFS 32
static float *g_bufs[MAX_BUFS];
static int g_nbufs = 0;

static float *tracked_tensor(GGUFHeader *hdr, const char *name,
                              uint64_t n_elems, float base,
                              uint64_t dim0, uint64_t dim1) {
    float *b = add_f32_tensor(hdr, name, n_elems, base, dim0, dim1);
    g_bufs[g_nbufs++] = b;
    return b;
}

static void free_tracked_bufs(void) {
    for (int i = 0; i < g_nbufs; i++) free(g_bufs[i]);
    g_nbufs = 0;
}

static void build_synthetic_header(GGUFHeader *hdr) {
    memset(hdr, 0, sizeof(*hdr));
    hdr->version = 3;
    strncpy(hdr->arch, "qwen3moe", sizeof(hdr->arch) - 1);

    add_meta_u32(hdr, "qwen3moe.embedding_length", T_DIM);
    add_meta_u32(hdr, "qwen3moe.block_count", T_LAYERS);
    add_meta_u32(hdr, "qwen3moe.attention.head_count", T_HEADS);
    add_meta_u32(hdr, "qwen3moe.attention.head_count_kv", T_KV_HEADS);
    add_meta_u32(hdr, "qwen3moe.feed_forward_length", T_HIDDEN);
    add_meta_u32(hdr, "qwen3moe.expert_count", T_EXPERTS);
    add_meta_u32(hdr, "qwen3moe.expert_used_count", T_EXPERTS_PER_TOK);
    add_meta_u32(hdr, "qwen3moe.expert_feed_forward_length", T_EXP_HID);
    add_meta_u32(hdr, "qwen3moe.attention.key_length", T_HEAD_DIM);

    /* token_embd.weight / output.weight: dims[0]=innermost=dim, dims[1]=vocab
     * — vocab_size is derived from this tensor's dims[1] since no explicit
     * qwen3moe.vocab_size meta key is set here (matches config_from_gguf's
     * documented fallback path). */
    tracked_tensor(hdr, "token_embd.weight", (uint64_t)T_VOCAB * T_DIM, 0.1f, T_DIM, T_VOCAB);
    tracked_tensor(hdr, "output.weight",     (uint64_t)T_VOCAB * T_DIM, 0.2f, T_DIM, T_VOCAB);
    tracked_tensor(hdr, "output_norm.weight", T_DIM, 1.0f, T_DIM, 0);

    tracked_tensor(hdr, "blk.0.attn_norm.weight", T_DIM, 1.0f, T_DIM, 0);
    tracked_tensor(hdr, "blk.0.ffn_norm.weight",  T_DIM, 1.0f, T_DIM, 0);
    tracked_tensor(hdr, "blk.0.attn_q_norm.weight", T_HEAD_DIM, 1.0f, T_HEAD_DIM, 0);
    tracked_tensor(hdr, "blk.0.attn_k_norm.weight", T_HEAD_DIM, 1.0f, T_HEAD_DIM, 0);

    tracked_tensor(hdr, "blk.0.attn_q.weight", (uint64_t)T_Q_ROWS  * T_DIM, 0.05f, T_DIM, T_Q_ROWS);
    tracked_tensor(hdr, "blk.0.attn_k.weight", (uint64_t)T_KV_ROWS * T_DIM, 0.05f, T_DIM, T_KV_ROWS);
    tracked_tensor(hdr, "blk.0.attn_v.weight", (uint64_t)T_KV_ROWS * T_DIM, 0.05f, T_DIM, T_KV_ROWS);
    tracked_tensor(hdr, "blk.0.attn_output.weight", (uint64_t)T_DIM * T_Q_ROWS, 0.05f, T_Q_ROWS, T_DIM);

    /* Router: dim -> num_experts, must be F32 (checked by the loader). */
    tracked_tensor(hdr, "blk.0.ffn_gate_inp.weight", (uint64_t)T_DIM * T_EXPERTS, 0.1f, T_DIM, T_EXPERTS);

    /* Stacked expert tensors: base + e-dependent-looking values via the
     * shared fill pattern already gives distinguishable per-expert regions
     * because each is a contiguous slice of one bigger tensor (stride =
     * exp_hid*dim floats per expert, exactly what the loader's stride-slice
     * arithmetic must reproduce). */
    tracked_tensor(hdr, "blk.0.ffn_gate_exps.weight",
                   (uint64_t)T_EXPERTS * T_EXP_HID * T_DIM, 1.0f, T_DIM, (uint64_t)T_EXP_HID * T_EXPERTS);
    tracked_tensor(hdr, "blk.0.ffn_up_exps.weight",
                   (uint64_t)T_EXPERTS * T_EXP_HID * T_DIM, 2.0f, T_DIM, (uint64_t)T_EXP_HID * T_EXPERTS);
    tracked_tensor(hdr, "blk.0.ffn_down_exps.weight",
                   (uint64_t)T_EXPERTS * T_EXP_HID * T_DIM, 3.0f, T_EXP_HID, (uint64_t)T_DIM * T_EXPERTS);
}

/* ── Q4_K expert-tensor fixture ──────────────────────────────────────────
 * Everything above builds expert tensors as F32, which never exercises
 * moe_ffn_forward()'s batched Q4_K path (moe_ffn.c: w->expert_w13_quant_type
 * == GGUF_TYPE_Q4_K) — the exact path a real Q4_K_M-quantized GGUF (the
 * only format anyone reports loading in practice, e.g. GitHub issue #32)
 * takes. That path had zero test coverage before this. Reuses the Q4_K
 * super-block encoder from test_q4k_matmul.c (own copy — tests don't share
 * translation units). */
#define TQ4K_SUPER 256
#define TQ4K_NSUB  8
#define TQ4K_BYTES 144

static uint16_t tq4k_f32_to_fp16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t sign  = (u >> 16) & 0x8000;
    uint32_t exp   = ((u >> 23) & 0xFF);
    uint32_t mant  = u & 0x7FFFFF;
    if (exp >= 143) return (uint16_t)(sign | 0x7BFF);
    if (exp <= 102) return (uint16_t)sign;
    return (uint16_t)(sign | ((exp - 112) << 10) | (mant >> 13));
}

static void tq4k_pack_scales(uint8_t *sc12, const uint8_t *sc, const uint8_t *mn) {
    for (int i = 0; i < 4; i++) {
        sc12[i]     = (uint8_t)((sc[i] & 0x3F) | (((sc[i + 4] >> 4) & 0x3) << 6));
        sc12[i + 4] = (uint8_t)((mn[i] & 0x3F) | (((mn[i + 4] >> 4) & 0x3) << 6));
    }
    for (int k = 0; k < 4; k++) {
        sc12[k + 8] = (uint8_t)((sc[k + 4] & 0xF) | ((mn[k + 4] & 0xF) << 4));
    }
}

/* Fills one valid (if arbitrary) 144-byte Q4_K super-block. Values don't
 * need to be numerically meaningful — this test is about crash/ASan
 * safety of the batched dispatch path, not output correctness (that's
 * test_q4k_matmul.c's job for the kernel itself). */
static void tq4k_make_block(uint8_t *blk, int seed) {
    float d    = 0.01f + (float)(seed % 11) * 0.011f;
    float dmin = 0.002f + (float)(seed % 7) * 0.003f;
    uint16_t d_h    = tq4k_f32_to_fp16(d);
    uint16_t dmin_h = tq4k_f32_to_fp16(dmin);
    memcpy(blk,     &d_h,    2);
    memcpy(blk + 2, &dmin_h, 2);

    uint8_t sc[TQ4K_NSUB], mn[TQ4K_NSUB];
    for (int j = 0; j < TQ4K_NSUB; j++) {
        sc[j] = (uint8_t)((seed * 7 + j * 5) % 64);
        mn[j] = (uint8_t)((seed * 3 + j * 11) % 64);
    }
    tq4k_pack_scales(blk + 4, sc, mn);

    uint8_t *qs = blk + 16;
    for (int i = 0; i < 128; i++) {
        int lo = (i * 5 + seed) & 0xF;
        int hi = (i * 7 + seed + 3) & 0xF;
        qs[i] = (uint8_t)(lo | (hi << 4));
    }
}

/* Builds one Q4_K-quantized "stacked experts" tensor: num_experts back-to-
 * back slices, each `rows` rows of `n_blocks_per_row` super-blocks —
 * exactly the row-major-per-expert layout weights_from_gguf_qwen3moe()'s
 * stride-slicing assumes (see gguf_loader.c: g_stride/u_stride/d_stride via
 * quant_bytes_for_elems(GGUF_TYPE_Q4_K, exp_hid*dim)). */
static uint8_t *tq4k_add_expert_tensor(GGUFHeader *hdr, const char *name,
                                        int num_experts, int rows,
                                        int n_blocks_per_row) {
    size_t row_bytes    = (size_t)n_blocks_per_row * TQ4K_BYTES;
    size_t expert_bytes = row_bytes * (size_t)rows;
    size_t total_bytes  = expert_bytes * (size_t)num_experts;

    uint8_t *buf = (uint8_t *)malloc(total_bytes);
    int seed = 1;
    for (size_t off = 0; off < total_bytes; off += TQ4K_BYTES) {
        tq4k_make_block(buf + off, seed++);
    }

    GGUFTensor *t = &hdr->tensors[hdr->n_tensors++];
    memset(t, 0, sizeof(*t));
    strncpy(t->name, name, GGUF_MAX_KEY_LEN - 1);
    t->type = GGUF_TYPE_Q4_K;
    t->data = buf;
    t->size_bytes = total_bytes;
    t->n_dims = 2;
    t->dims[0] = (uint64_t)(n_blocks_per_row * TQ4K_SUPER);
    t->dims[1] = (uint64_t)(rows * num_experts);
    return buf;
}

#define TQ_DIM        512
#define TQ_LAYERS     1
#define TQ_HEADS      4
#define TQ_KV_HEADS   2
#define TQ_HEAD_DIM   32
#define TQ_VOCAB      6
#define TQ_EXPERTS    128
#define TQ_EXPERTS_PER_TOK 8
#define TQ_EXP_HID    256

#define TQ_Q_ROWS  (TQ_HEADS    * TQ_HEAD_DIM)
#define TQ_KV_ROWS (TQ_KV_HEADS * TQ_HEAD_DIM)

static void build_synthetic_header_q4k_experts(GGUFHeader *hdr, uint8_t **out_bufs, int *out_n) {
    memset(hdr, 0, sizeof(*hdr));
    hdr->version = 3;
    strncpy(hdr->arch, "qwen3moe", sizeof(hdr->arch) - 1);

    add_meta_u32(hdr, "qwen3moe.embedding_length", TQ_DIM);
    add_meta_u32(hdr, "qwen3moe.block_count", TQ_LAYERS);
    add_meta_u32(hdr, "qwen3moe.attention.head_count", TQ_HEADS);
    add_meta_u32(hdr, "qwen3moe.attention.head_count_kv", TQ_KV_HEADS);
    add_meta_u32(hdr, "qwen3moe.feed_forward_length", TQ_DIM);
    add_meta_u32(hdr, "qwen3moe.expert_count", TQ_EXPERTS);
    add_meta_u32(hdr, "qwen3moe.expert_used_count", TQ_EXPERTS_PER_TOK);
    add_meta_u32(hdr, "qwen3moe.expert_feed_forward_length", TQ_EXP_HID);
    add_meta_u32(hdr, "qwen3moe.attention.key_length", TQ_HEAD_DIM);

    tracked_tensor(hdr, "token_embd.weight", (uint64_t)TQ_VOCAB * TQ_DIM, 0.1f, TQ_DIM, TQ_VOCAB);
    tracked_tensor(hdr, "output.weight",     (uint64_t)TQ_VOCAB * TQ_DIM, 0.2f, TQ_DIM, TQ_VOCAB);
    tracked_tensor(hdr, "output_norm.weight", TQ_DIM, 1.0f, TQ_DIM, 0);

    tracked_tensor(hdr, "blk.0.attn_norm.weight", TQ_DIM, 1.0f, TQ_DIM, 0);
    tracked_tensor(hdr, "blk.0.ffn_norm.weight",  TQ_DIM, 1.0f, TQ_DIM, 0);
    tracked_tensor(hdr, "blk.0.attn_q_norm.weight", TQ_HEAD_DIM, 1.0f, TQ_HEAD_DIM, 0);
    tracked_tensor(hdr, "blk.0.attn_k_norm.weight", TQ_HEAD_DIM, 1.0f, TQ_HEAD_DIM, 0);

    tracked_tensor(hdr, "blk.0.attn_q.weight", (uint64_t)TQ_Q_ROWS  * TQ_DIM, 0.05f, TQ_DIM, TQ_Q_ROWS);
    tracked_tensor(hdr, "blk.0.attn_k.weight", (uint64_t)TQ_KV_ROWS * TQ_DIM, 0.05f, TQ_DIM, TQ_KV_ROWS);
    tracked_tensor(hdr, "blk.0.attn_v.weight", (uint64_t)TQ_KV_ROWS * TQ_DIM, 0.05f, TQ_DIM, TQ_KV_ROWS);
    tracked_tensor(hdr, "blk.0.attn_output.weight", (uint64_t)TQ_DIM * TQ_Q_ROWS, 0.05f, TQ_Q_ROWS, TQ_DIM);

    tracked_tensor(hdr, "blk.0.ffn_gate_inp.weight", (uint64_t)TQ_DIM * TQ_EXPERTS, 0.1f, TQ_DIM, TQ_EXPERTS);

    /* Real Q4_K expert tensors — the untested path. gate/up: expert_hdim
     * rows x (dim/256) blocks/row. down: dim rows x (expert_hdim/256)
     * blocks/row. Both total exp_hid*dim/256 blocks per expert, matching
     * quant_bytes_for_elems()'s stride formula in gguf_loader.c. */
    int n = 0;
    out_bufs[n++] = tq4k_add_expert_tensor(hdr, "blk.0.ffn_gate_exps.weight",
                                            TQ_EXPERTS, TQ_EXP_HID, TQ_DIM / TQ4K_SUPER);
    out_bufs[n++] = tq4k_add_expert_tensor(hdr, "blk.0.ffn_up_exps.weight",
                                            TQ_EXPERTS, TQ_EXP_HID, TQ_DIM / TQ4K_SUPER);
    out_bufs[n++] = tq4k_add_expert_tensor(hdr, "blk.0.ffn_down_exps.weight",
                                            TQ_EXPERTS, TQ_DIM, TQ_EXP_HID / TQ4K_SUPER);
    *out_n = n;
}

static void test_moe_ffn_q4k_batched_path(void) {
    GGUFHeader *hdr = (GGUFHeader *)malloc(sizeof(GGUFHeader));
    uint8_t *q4k_bufs[8];
    int n_q4k_bufs = 0;
    build_synthetic_header_q4k_experts(hdr, q4k_bufs, &n_q4k_bufs);

    Config cfg;
    MoEConfig mc;
    TransformerWeights w;
    GGUFWeightStore *store = NULL;
    RunState s;
    int rs_alloc_ok = 0, qwen3moe_rs_alloc_ok = 0, w_alloc_ok = 0, moe_alloc_ok = 0;
    ThreadPool *tp = NULL;

    if (config_from_gguf(&cfg, hdr) != TN_OK) { FAIL("moe_ffn_q4k_batched", "config_from_gguf failed"); goto cleanup; }
    if (moe_config_from_gguf(&mc, hdr) != TN_OK) { FAIL("moe_ffn_q4k_batched", "moe_config_from_gguf failed"); goto cleanup; }
    if (mc.num_experts != TQ_EXPERTS || mc.num_experts_per_tok != TQ_EXPERTS_PER_TOK) {
        FAIL("moe_ffn_q4k_batched", "MoEConfig expert counts mismatch"); goto cleanup;
    }

    memset(&w, 0, sizeof(w));
    if (weights_alloc_pointers(&w, &cfg) != TN_OK) { FAIL("moe_ffn_q4k_batched", "weights_alloc_pointers failed"); goto cleanup; }
    w_alloc_ok = 1;
    if (moe_weights_alloc(&w, &cfg, &mc) != TN_OK) { FAIL("moe_ffn_q4k_batched", "moe_weights_alloc failed"); goto cleanup; }
    moe_alloc_ok = 1;
    if (weights_from_gguf(&w, &cfg, hdr, &store) != TN_OK) { FAIL("moe_ffn_q4k_batched", "weights_from_gguf failed"); goto cleanup; }

    if (w.expert_w13_quant_type != GGUF_TYPE_Q4_K) {
        FAIL("moe_ffn_q4k_batched", "expert_w13_quant_type != Q4_K -- fixture didn't hit the batched path"); goto cleanup;
    }

    cfg.seq_len = 16;
    if (run_state_alloc_ex(&s, &cfg, cfg.seq_len, mc.has_qk_norm) != TN_OK) {
        FAIL("moe_ffn_q4k_batched", "run_state_alloc_ex failed"); goto cleanup;
    }
    rs_alloc_ok = 1;
    if (qwen3moe_run_state_alloc(&s, &cfg, &mc, cfg.seq_len) != TN_OK) {
        FAIL("moe_ffn_q4k_batched", "qwen3moe_run_state_alloc failed"); goto cleanup;
    }
    qwen3moe_rs_alloc_ok = 1;

    /* Real usage always runs the batched Q4K matmul through a real
     * ThreadPool (main.c: 9 active threads in the reported crash) —
     * matmul_q4k_batch_task then executes via threadpool_dispatch's
     * chunked start/end ranges rather than the single-shot [0,k*d) call
     * a NULL tp takes, so a race or chunking bug wouldn't show with
     * tp=NULL. */
    tp = threadpool_create(8);

    /* Several tokens/positions, to also exercise the KV-cache write path
     * (mapped_pos, sliding window bookkeeping) across more than pos=0. */
    for (int pos = 0; pos < 4; pos++) {
        float *logits = transformer_forward(pos % TQ_VOCAB, pos, &cfg, &w, &s, &mc, tp);
        if (!logits) { FAIL("moe_ffn_q4k_batched", "transformer_forward returned NULL"); goto cleanup; }
        for (int i = 0; i < cfg.vocab_size; i++) {
            if (!isfinite(logits[i])) {
                FAIL("moe_ffn_q4k_batched", "logits contain non-finite value"); goto cleanup;
            }
        }
    }
    PASS("moe_ffn_q4k_batched_path");

cleanup:
    if (tp) threadpool_destroy(tp);
    if (qwen3moe_rs_alloc_ok) qwen3moe_run_state_free(&s, &cfg);
    if (rs_alloc_ok) run_state_free(&s);
    if (store) weights_free_gguf(store);
    if (moe_alloc_ok) moe_weights_free(&w, &mc);
    if (w_alloc_ok) weights_free_pointers(&w);
    free_tracked_bufs();
    for (int i = 0; i < n_q4k_bufs; i++) free(q4k_bufs[i]);
    free(hdr);
}

static void test_config_and_moe_config(void) {
    GGUFHeader *hdr = (GGUFHeader *)malloc(sizeof(GGUFHeader));
    build_synthetic_header(hdr);

    Config cfg;
    if (config_from_gguf(&cfg, hdr) != TN_OK) {
        FAIL("config_and_moe_config", "config_from_gguf failed"); goto done;
    }
    if (cfg.dim != T_DIM || cfg.n_layers != T_LAYERS || cfg.n_heads != T_HEADS ||
        cfg.n_kv_heads != T_KV_HEADS || cfg.vocab_size != T_VOCAB) {
        FAIL("config_and_moe_config", "Config fields mismatch"); goto done;
    }

    MoEConfig mc;
    if (moe_config_from_gguf(&mc, hdr) != TN_OK) {
        FAIL("config_and_moe_config", "moe_config_from_gguf failed"); goto done;
    }
    if (!mc.is_moe || mc.has_mla != 0 || mc.has_qk_norm != 1 ||
        mc.n_shared_experts != 0 || mc.first_k_dense_replace != 0 ||
        mc.num_experts != T_EXPERTS || mc.num_experts_per_tok != T_EXPERTS_PER_TOK ||
        mc.expert_hidden_dim != T_EXP_HID || mc.attn_head_dim != T_HEAD_DIM) {
        FAIL("config_and_moe_config", "MoEConfig fields mismatch"); goto done;
    }
    PASS("config_and_moe_config");

done:
    free_tracked_bufs();
    free(hdr);
}

static void test_weights_from_gguf_and_expert_stride(void) {
    GGUFHeader *hdr = (GGUFHeader *)malloc(sizeof(GGUFHeader));
    build_synthetic_header(hdr);

    Config cfg;
    MoEConfig mc;
    TransformerWeights w;
    GGUFWeightStore *store = NULL;

    if (config_from_gguf(&cfg, hdr) != TN_OK) { FAIL("weights_from_gguf", "config_from_gguf failed"); goto done_hdr; }
    if (moe_config_from_gguf(&mc, hdr) != TN_OK) { FAIL("weights_from_gguf", "moe_config_from_gguf failed"); goto done_hdr; }

    memset(&w, 0, sizeof(w));
    if (weights_alloc_pointers(&w, &cfg) != TN_OK) { FAIL("weights_from_gguf", "weights_alloc_pointers failed"); goto done_hdr; }
    if (moe_weights_alloc(&w, &cfg, &mc) != TN_OK) { FAIL("weights_from_gguf", "moe_weights_alloc failed"); goto done_w; }

    if (weights_from_gguf(&w, &cfg, hdr, &store) != TN_OK) {
        FAIL("weights_from_gguf", "weights_from_gguf returned non-TN_OK"); goto done_moe;
    }

    if (!w.qwen3moe_attn_q_norm || !w.qwen3moe_attn_k_norm) {
        FAIL("weights_from_gguf", "expected qwen3moe_attn_q_norm/k_norm arrays to be allocated"); goto done_store;
    }
    if (!w.qwen3moe_attn_q_norm[0] || !w.qwen3moe_attn_k_norm[0]) {
        FAIL("weights_from_gguf", "qwen3moe_attn_q_norm/k_norm[0] is NULL"); goto done_store;
    }
    if (!w.moe_gate_w[0]) {
        FAIL("weights_from_gguf", "moe_gate_w[0] is NULL"); goto done_store;
    }
    if (!w.moe_w1[0][0] || !w.moe_w1[0][1]) {
        FAIL("weights_from_gguf", "moe_w1[0][0 or 1] is NULL"); goto done_store;
    }
    /* Per-expert stride correctness: expert 0 and expert 1 must be distinct
     * offsets into the stacked tensor, not aliasing the same base pointer
     * (the exact bug class this stride-slicing scheme is prone to). */
    if (w.moe_w1[0][0] == w.moe_w1[0][1]) {
        FAIL("weights_from_gguf", "moe_w1[0][0] aliases moe_w1[0][1] (stride bug)"); goto done_store;
    }
    ptrdiff_t stride_bytes = (const tn_i8 *)w.moe_w1[0][1] - (const tn_i8 *)w.moe_w1[0][0];
    ptrdiff_t expected_stride = (ptrdiff_t)((size_t)T_EXP_HID * T_DIM * sizeof(float));
    if (stride_bytes != expected_stride) {
        FAIL("weights_from_gguf", "expert stride does not match expert_hidden_dim*dim*sizeof(float)");
        goto done_store;
    }
    PASS("weights_from_gguf_and_expert_stride");

done_store:
    weights_free_gguf(store);
done_moe:
    moe_weights_free(&w, &mc);
done_w:
    weights_free_pointers(&w);
done_hdr:
    free_tracked_bufs();
    free(hdr);
}

static void test_end_to_end_forward_pass(void) {
    GGUFHeader *hdr = (GGUFHeader *)malloc(sizeof(GGUFHeader));
    build_synthetic_header(hdr);

    Config cfg;
    MoEConfig mc;
    TransformerWeights w;
    GGUFWeightStore *store = NULL;
    RunState s;
    int rs_alloc_ok = 0, qwen3moe_rs_alloc_ok = 0, w_alloc_ok = 0, moe_alloc_ok = 0;

    if (config_from_gguf(&cfg, hdr) != TN_OK) { FAIL("end_to_end_forward", "config_from_gguf failed"); goto cleanup; }
    if (moe_config_from_gguf(&mc, hdr) != TN_OK) { FAIL("end_to_end_forward", "moe_config_from_gguf failed"); goto cleanup; }

    memset(&w, 0, sizeof(w));
    if (weights_alloc_pointers(&w, &cfg) != TN_OK) { FAIL("end_to_end_forward", "weights_alloc_pointers failed"); goto cleanup; }
    w_alloc_ok = 1;
    if (moe_weights_alloc(&w, &cfg, &mc) != TN_OK) { FAIL("end_to_end_forward", "moe_weights_alloc failed"); goto cleanup; }
    moe_alloc_ok = 1;
    if (weights_from_gguf(&w, &cfg, hdr, &store) != TN_OK) { FAIL("end_to_end_forward", "weights_from_gguf failed"); goto cleanup; }

    cfg.seq_len = 16; /* small context, plenty for a 1-token forward pass */
    if (run_state_alloc_ex(&s, &cfg, cfg.seq_len, mc.has_qk_norm) != TN_OK) {
        FAIL("end_to_end_forward", "run_state_alloc_ex failed"); goto cleanup;
    }
    rs_alloc_ok = 1;
    if (qwen3moe_run_state_alloc(&s, &cfg, &mc, cfg.seq_len) != TN_OK) {
        FAIL("end_to_end_forward", "qwen3moe_run_state_alloc failed"); goto cleanup;
    }
    qwen3moe_rs_alloc_ok = 1;

    float *logits = transformer_forward(0 /* token */, 0 /* pos */, &cfg, &w, &s, &mc, NULL /* tp */);
    if (!logits) { FAIL("end_to_end_forward", "transformer_forward returned NULL"); goto cleanup; }

    for (int i = 0; i < cfg.vocab_size; i++) {
        if (!isfinite(logits[i])) {
            FAIL("end_to_end_forward", "logits contain non-finite value"); goto cleanup;
        }
    }
    PASS("end_to_end_forward");

cleanup:
    if (qwen3moe_rs_alloc_ok) qwen3moe_run_state_free(&s, &cfg);
    if (rs_alloc_ok) run_state_free(&s);
    if (store) weights_free_gguf(store);
    if (moe_alloc_ok) moe_weights_free(&w, &mc);
    if (w_alloc_ok) weights_free_pointers(&w);
    free_tracked_bufs();
    free(hdr);
}

int main(void) {
    tn_simd_init(); /* tn_rmsnorm/tn_vec_dot/etc. are NULL function pointers
                      * until this runs — every real entrypoint (main.c) calls
                      * it before any forward pass; this test must too. */
    printf("=== Qwen3-MoE GGUF Loader Tests ===\n");
    test_config_and_moe_config();
    test_weights_from_gguf_and_expert_stride();
    test_end_to_end_forward_pass();
    test_moe_ffn_q4k_batched_path();

    printf("\n");
    if (g_failures == 0) { printf("=== All qwen3moe loader tests passed ===\n"); return 0; }
    printf("=== %d test(s) FAILED ===\n", g_failures);
    return 1;
}
