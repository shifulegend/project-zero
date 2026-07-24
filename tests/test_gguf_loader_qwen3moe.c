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

    printf("\n");
    if (g_failures == 0) { printf("=== All qwen3moe loader tests passed ===\n"); return 0; }
    printf("=== %d test(s) FAILED ===\n", g_failures);
    return 1;
}
