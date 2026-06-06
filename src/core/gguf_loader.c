#include "core/gguf_loader.h"
#include "core/gguf_quant.h"
#include "core/moe_config.h"
#include "core/platform.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ── GGUFWeightStore — growable list of heap buffers to free on cleanup ──── */

struct GGUFWeightStore {
    void **bufs;
    size_t count;
    size_t cap;
};

static GGUFWeightStore *store_alloc(void) {
    GGUFWeightStore *s = (GGUFWeightStore *)malloc(sizeof(GGUFWeightStore));
    if (!s) return NULL;
    s->cap   = 64;
    s->count = 0;
    s->bufs  = (void **)malloc(s->cap * sizeof(void *));
    if (!s->bufs) { free(s); return NULL; }
    return s;
}

static int store_add(GGUFWeightStore *s, void *ptr) {
    if (s->count == s->cap) {
        size_t new_cap = s->cap * 2;
        void **nb = (void **)realloc(s->bufs, new_cap * sizeof(void *));
        if (!nb) return -1;
        s->bufs = nb;
        s->cap  = new_cap;
    }
    s->bufs[s->count++] = ptr;
    return 0;
}

void weights_free_gguf(GGUFWeightStore *s) {
    if (!s) return;
    for (size_t i = 0; i < s->count; i++) free(s->bufs[i]);
    free(s->bufs);
    free(s);
}

/* ── Dequantization helpers ───────────────────────────────────────────────── */

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = (uint32_t)(h & 0x3ff) << 13;
    uint32_t bits;
    if      (exp == 0)  bits = sign | mant;               /* subnormal / zero */
    else if (exp == 31) bits = sign | 0x7f800000u | mant; /* inf / NaN       */
    else                bits = sign | ((exp + 112) << 23) | mant;
    float f; memcpy(&f, &bits, 4); return f;
}

static float bf16_to_f32(uint16_t b) {
    uint32_t bits = (uint32_t)b << 16;
    float f; memcpy(&f, &bits, 4); return f;
}

/* Allocate a float32 buffer (n_elems floats), fill from tensor data, add to store.
 * Returns NULL and prints a message for unsupported types. */
static float *tensor_to_f32(const GGUFTensor *t, size_t n_elems,
                              GGUFWeightStore *store) {
    float *buf = (float *)malloc(n_elems * sizeof(float));
    if (!buf) return NULL;

    switch (t->type) {
    case GGUF_TYPE_F32:
        memcpy(buf, t->data, n_elems * sizeof(float));
        break;
    case GGUF_TYPE_F16: {
        const uint16_t *s = (const uint16_t *)t->data;
        for (size_t i = 0; i < n_elems; i++) buf[i] = f16_to_f32(s[i]);
        break;
    }
    case GGUF_TYPE_BF16: {
        const uint16_t *s = (const uint16_t *)t->data;
        for (size_t i = 0; i < n_elems; i++) buf[i] = bf16_to_f32(s[i]);
        break;
    }
    case GGUF_TYPE_Q8_0:
        gguf_dequant_q8_0(buf, t->data, n_elems);
        break;
    case GGUF_TYPE_Q4_K:
        gguf_dequant_q4_k(buf, t->data, n_elems);
        break;
    case GGUF_TYPE_Q4_0:
        gguf_dequant_q4_0(buf, t->data, n_elems);
        break;
    case GGUF_TYPE_Q5_0:
        gguf_dequant_q5_0(buf, t->data, n_elems);
        break;
    case GGUF_TYPE_Q5_1:
        gguf_dequant_q5_1(buf, t->data, n_elems);
        break;
    case GGUF_TYPE_Q5_K:
        gguf_dequant_q5_k(buf, t->data, n_elems);
        break;
    case GGUF_TYPE_Q6_K:
        gguf_dequant_q6_k(buf, t->data, n_elems);
        break;
    case GGUF_TYPE_Q2_K:
        gguf_dequant_q2_k(buf, t->data, n_elems);
        break;
    case GGUF_TYPE_Q3_K:
        gguf_dequant_q3_k(buf, t->data, n_elems);
        break;
    case GGUF_TYPE_IQ4_NL:
        gguf_dequant_iq4_nl(buf, t->data, n_elems);
        break;
    default:
        fprintf(stderr,
            "[gguf_loader] unsupported quant type %d ('%s') for tensor '%s'\n"
            "[gguf_loader] Supported: F32, F16, BF16, Q8_0, Q4_K, Q4_0, Q5_0, Q5_1, Q5_K, Q6_K, Q2_K, Q3_K, IQ4_NL\n"
            "[gguf_loader] Convert with: llama-quantize model.gguf out.gguf Q4_K_S\n",
            (int)t->type, gguf_type_name(t->type), t->name);
        free(buf);
        return NULL;
    }

    if (store_add(store, buf) != 0) { free(buf); return NULL; }
    return buf;
}

/* ── Norm-weight helper: F32 → zero-copy; F16/BF16 → dequant into store ─── */

static float *norm_to_f32(const GGUFHeader *hdr, const char *name,
                            size_t n_elems, GGUFWeightStore *store) {
    const GGUFTensor *t = gguf_find_tensor(hdr, name);
    if (!t) {
        fprintf(stderr, "[gguf_loader] missing norm tensor '%s'\n", name);
        return NULL;
    }
    if (t->type == GGUF_TYPE_F32) {
        return (float *)t->data; /* zero-copy from mmap */
    }
    return tensor_to_f32(t, n_elems, store);
}

/* ── config_from_gguf() ───────────────────────────────────────────────────── */

TernaryError config_from_gguf(Config *cfg, const GGUFHeader *hdr) {
    memset(cfg, 0, sizeof(*cfg));

    /* Architecture prefix — "llama" for most models, "deepseek2" for DeepSeek */
    const char *arch = hdr->arch;
    char emb_key[128], blk_key[128], head_key[128], ffn_key[128];
    char kv_head_key[128], ctx_key[128], vocab_key[128], rope_key[128];
    snprintf(emb_key,    sizeof(emb_key),    "%s.embedding_length",       arch);
    snprintf(blk_key,    sizeof(blk_key),    "%s.block_count",             arch);
    snprintf(head_key,   sizeof(head_key),   "%s.attention.head_count",    arch);
    snprintf(ffn_key,    sizeof(ffn_key),    "%s.feed_forward_length",     arch);
    snprintf(kv_head_key,sizeof(kv_head_key),"%s.attention.head_count_kv", arch);
    snprintf(ctx_key,    sizeof(ctx_key),    "%s.context_length",          arch);
    snprintf(vocab_key,  sizeof(vocab_key),  "%s.vocab_size",              arch);
    snprintf(rope_key,   sizeof(rope_key),   "%s.rope.freq_base",          arch);

    cfg->dim       = (int)gguf_meta_u32(hdr, emb_key,  0);
    cfg->n_layers  = (int)gguf_meta_u32(hdr, blk_key,  0);
    cfg->n_heads   = (int)gguf_meta_u32(hdr, head_key, 0);
    cfg->hidden_dim= (int)gguf_meta_u32(hdr, ffn_key,  0);

    if (cfg->dim <= 0 || cfg->n_layers <= 0 ||
        cfg->n_heads <= 0 || cfg->hidden_dim <= 0) {
        fprintf(stderr,
            "[gguf_loader] missing required metadata (embedding_length / "
            "block_count / head_count / feed_forward_length)\n"
            "[gguf_loader] Architecture in file: '%s'\n", arch);
        return TN_ERR_INVALID_CONFIG;
    }

    cfg->n_kv_heads = (int)gguf_meta_u32(hdr, kv_head_key, (uint32_t)cfg->n_heads);
    cfg->seq_len    = (int)gguf_meta_u32(hdr, ctx_key,     4096);
    cfg->vocab_size = (int)gguf_meta_u32(hdr, vocab_key,   0);

    /* Vocab size fallback: read from token_embd.weight tensor (dims[1] = rows) */
    if (cfg->vocab_size <= 0) {
        const GGUFTensor *embd = gguf_find_tensor(hdr, "token_embd.weight");
        if (embd && embd->n_dims >= 2)
            cfg->vocab_size = (int)embd->dims[1];
    }
    if (cfg->vocab_size <= 0) {
        fprintf(stderr, "[gguf_loader] cannot determine vocab_size\n");
        return TN_ERR_INVALID_CONFIG;
    }

    /* rope_theta: stored as FLOAT32 in GGUF, not uint32 — scan meta manually */
    cfg->rope_theta = 10000.0f;
    for (uint64_t i = 0; i < hdr->n_meta; i++) {
        if (strcmp(hdr->meta[i].key, rope_key) == 0) {
            if (hdr->meta[i].val_type == GGUF_VAL_FLOAT32)
                cfg->rope_theta = hdr->meta[i].val.f32;
            else if (hdr->meta[i].val_type == GGUF_VAL_UINT32)
                cfg->rope_theta = (float)hdr->meta[i].val.u32;
            break;
        }
    }

    cfg->scale_mode    = 1; /* non-ternary float path */
    cfg->act_type      = 0; /* SiLU */
    cfg->bos_token_id  = (int)gguf_meta_u32(hdr, "tokenizer.ggml.bos_token_id", (uint32_t)-1);
    cfg->eos_token_id  = (int)gguf_meta_u32(hdr, "tokenizer.ggml.eos_token_id", (uint32_t)-1);

    /* RMSNorm epsilon — DeepSeek uses 1e-6, most others use 1e-5 */
    {
        char eps_key[128];
        snprintf(eps_key, 128, "%s.attention.layer_norm_rms_epsilon", hdr->arch);
        cfg->rms_norm_eps = gguf_meta_f32(hdr, eps_key, 1e-5f);
        if (cfg->rms_norm_eps <= 0.0f || cfg->rms_norm_eps > 1e-3f)
            cfg->rms_norm_eps = 1e-5f; /* sanity clamp */
    }

    /* ── YaRN RoPE scaling parameters ─────────────────────────────────── */
    /* Build arch-prefixed keys for rope.scaling.* */
    char scale_key[128], scale_type_key[128], log_mul_key[128], orig_ctx_key[128];
    snprintf(scale_key,     128,     "%s.rope.scaling.factor",                   hdr->arch);
    snprintf(scale_type_key,128, "%s.rope.scaling.type",                    hdr->arch);
    snprintf(log_mul_key,   128,    "%s.rope.scaling.yarn_log_multiplier",      hdr->arch);
    snprintf(orig_ctx_key,  128,   "%s.rope.scaling.original_context_length",  hdr->arch);

    float rope_scale_factor = gguf_meta_f32(hdr, scale_key, 1.0f);
    const char *scale_type  = gguf_meta_str(hdr, scale_type_key);
    float yarn_log_mul      = gguf_meta_f32(hdr, log_mul_key, 0.0f);
    int   orig_ctx          = (int)gguf_meta_u32(hdr, orig_ctx_key, (uint32_t)cfg->seq_len);

    /* freq_scale = 1/factor; ext_factor=1 for yarn scaling, 0 otherwise */
    cfg->rope_freq_scale      = (rope_scale_factor > 1.0f) ? (1.0f / rope_scale_factor) : 1.0f;
    cfg->rope_yarn_ext_factor = (scale_type && strcmp(scale_type, "yarn") == 0) ? 1.0f : 0.0f;
    cfg->rope_orig_ctx_len    = (orig_ctx > 0) ? orig_ctx : cfg->seq_len;
    cfg->rope_yarn_beta_fast  = 32.0f;
    cfg->rope_yarn_beta_slow  = 1.0f;
    /* Store raw yarn_log_multiplier for kq_scale computation in attention.
     * Used as: mscale_kq = 1 + rope_yarn_log_mul * ln(factor)
     * then:    iscale = mscale_kq^2 / sqrt(head_dim)  (matches llama.cpp deepseek2.cpp) */
    cfg->rope_yarn_log_mul = yarn_log_mul;

    /* rope_yarn_attn_factor is the RoPE amplitude passed to ggml_rope_ext.
     * In our rope code: mscale = attn_factor * (1 + 0.1 * ln(1/freq_scale))
     * To match llama.cpp (RoPE mscale = 1.0 for DeepSeek YaRN):
     *   attn_factor = 1 / (1 + 0.1 * ln(factor))
     * so that: attn_factor * (1 + 0.1 * ln(factor)) = 1.0
     * This cancels out the YaRN correction and applies no amplitude scaling to RoPE,
     * consistent with llama.cpp's deepseek2 approach where kq_scale carries mscale^2. */
    if (cfg->rope_yarn_ext_factor != 0.0f && rope_scale_factor > 1.0f) {
        cfg->rope_yarn_attn_factor = 1.0f / (1.0f + 0.1f * logf(rope_scale_factor));
    } else {
        cfg->rope_yarn_attn_factor = 1.0f;
    }

    return TN_OK;
}

/* ── moe_config_from_gguf() ───────────────────────────────────────────────── */

TernaryError moe_config_from_gguf(MoEConfig *mc, const GGUFHeader *hdr) {
    moe_config_init_dense(mc);
    if (strcmp(hdr->arch, "deepseek2") != 0) return TN_OK;

    /* All keys use the model architecture prefix from the GGUF header */
    char k[128];
#define MK(suffix) (snprintf(k, sizeof(k), "%s." suffix, hdr->arch), k)

    mc->num_experts         = (int)gguf_meta_u32(hdr, MK("expert_count"),               64);
    mc->num_experts_per_tok = (int)gguf_meta_u32(hdr, MK("expert_used_count"),            6);
    mc->expert_hidden_dim   = (int)gguf_meta_u32(hdr, MK("expert_feed_forward_length"), 1408);
    mc->first_k_dense_replace=(int)gguf_meta_u32(hdr, MK("leading_dense_block_count"),   1);
    mc->n_shared_experts    = (int)gguf_meta_u32(hdr, MK("expert_shared_count"),          2);
    mc->kv_lora_rank        = (int)gguf_meta_u32(hdr, MK("attention.kv_lora_rank"),     512);
    mc->qk_rope_head_dim    = (int)gguf_meta_u32(hdr, MK("rope.dimension_count"),        64);
    mc->v_head_dim          = (int)gguf_meta_u32(hdr, MK("attention.value_length"),     128);

    /* qk_nope = total_key_length - rope_dim */
    int key_length          = (int)gguf_meta_u32(hdr, MK("attention.key_length"),       192);
    mc->qk_nope_head_dim    = key_length - mc->qk_rope_head_dim;
    if (mc->qk_nope_head_dim <= 0) mc->qk_nope_head_dim = 128; /* fallback */

#undef MK

    /* shared hidden = n_shared * expert_hidden */
    mc->shared_expert_hidden_dim = mc->n_shared_experts * mc->expert_hidden_dim;

    mc->is_moe  = true;
    mc->has_mla = 1;
    return TN_OK;
}

/* ── Byte-count helper for quantised blocks ──────────────────────────────── */

static size_t quant_bytes_for_elems(GGUFType type, size_t n_elems) {
    switch (type) {
    case GGUF_TYPE_F32:  return n_elems * 4;
    case GGUF_TYPE_F16:  return n_elems * 2;
    case GGUF_TYPE_BF16: return n_elems * 2;
    case GGUF_TYPE_Q8_0: return (n_elems / 32)  * 34;
    case GGUF_TYPE_Q4_0: return (n_elems / 32)  * 18;
    case GGUF_TYPE_Q5_0: return (n_elems / 32)  * 22;
    case GGUF_TYPE_Q5_1: return (n_elems / 32)  * 24;
    case GGUF_TYPE_Q4_K: return (n_elems / 256) * 144;
    case GGUF_TYPE_Q5_K: return (n_elems / 256) * 176;
    case GGUF_TYPE_Q6_K: return (n_elems / 256) * 210;
    case GGUF_TYPE_Q2_K: return (n_elems / 256) * 84;
    case GGUF_TYPE_Q3_K: return (n_elems / 256) * 110;
    case GGUF_TYPE_IQ4_NL: return (n_elems / 32) * 18;
    default: return 0;
    }
}

/* ── weights_from_gguf_deepseek2() — DeepSeek-V2 MLA+MoE weight loader ─── */

static TernaryError weights_from_gguf_deepseek2(
        TransformerWeights *w, const Config *cfg, const GGUFHeader *hdr,
        const MoEConfig *mc, GGUFWeightStore *store) {

    int dim        = cfg->dim;
    int hidden_dim = cfg->hidden_dim;
    int nl         = cfg->n_layers;
    int n_heads    = cfg->n_heads;
    int n_kv_heads = cfg->n_kv_heads;
    int first_dense= mc->first_k_dense_replace;
    int num_experts= mc->num_experts;
    int exp_hid    = mc->expert_hidden_dim;
    int sh_hid     = mc->shared_expert_hidden_dim;
    int lora       = mc->kv_lora_rank;
    int nope       = mc->qk_nope_head_dim;
    int rope_dim   = mc->qk_rope_head_dim;
    int v_dim      = mc->v_head_dim;
    char name_buf[128];

    /* MLA dimension sizes */
    int q_rows   = n_heads  * (nope + rope_dim);  /* e.g. 16*192=3072 */
    int kva_rows = lora + rope_dim;               /* e.g. 576          */
    int kvb_rows = n_kv_heads * (nope + v_dim);  /* e.g. 16*256=4096  */

#define DS_LOAD_PROJ(field, tname, n_elems) do {                            \
    snprintf(name_buf, sizeof(name_buf), "blk.%d." tname, l);              \
    const GGUFTensor *_t = gguf_find_tensor(hdr, name_buf);                \
    if (!_t) {                                                              \
        fprintf(stderr, "[gguf_loader] missing tensor '%s'\n", name_buf);  \
        return TN_ERR_INVALID_WEIGHTS;                                      \
    }                                                                       \
    float *_f = tensor_to_f32(_t, (n_elems), store);                       \
    if (!_f) return TN_ERR_INVALID_WEIGHTS;                                 \
    (field)[l] = (tn_i8 *)_f;                                              \
} while(0)

/* Zero-copy loader for MLA projection weights (mmap pointer, may page-fault).
 * Falls back to F32 dequant for non-Q4_K types. */
#define DS_LOAD_MLA_PROJ(field, tname, n_elems) do {                       \
    snprintf(name_buf, sizeof(name_buf), "blk.%d." tname, l);              \
    const GGUFTensor *_t = gguf_find_tensor(hdr, name_buf);                \
    if (!_t) {                                                              \
        fprintf(stderr, "[gguf_loader] missing tensor '%s'\n", name_buf);  \
        return TN_ERR_INVALID_WEIGHTS;                                      \
    }                                                                       \
    if (_t->type == GGUF_TYPE_Q4_K) {                                      \
        (field)[l] = (tn_i8 *)_t->data; /* zero-copy raw Q4_K bytes */     \
        w->has_mla_quant = true;                                            \
    } else {                                                                \
        float *_f = tensor_to_f32(_t, (n_elems), store);                   \
        if (!_f) return TN_ERR_INVALID_WEIGHTS;                             \
        (field)[l] = (tn_i8 *)_f;                                          \
    }                                                                       \
} while(0)

/* Heap-copy loader for shared expert weights.
 * Stores raw Q4_K bytes when tensor is Q4_K (type 12), else dequantizes to F32.
 * Records the actual GGUF type per-layer in type_arr[l] for correct dispatch.
 * This avoids the global-flag pitfall where mixed-type layers cause wrong kernel dispatch
 * (e.g. ffn_down_shexp is Q5_K on layer 1 but Q4_K on all other layers). */
#define DS_LOAD_SHEXP_HEAP(field, tname, n_elems, type_arr) do {            \
    snprintf(name_buf, sizeof(name_buf), "blk.%d." tname, l);              \
    const GGUFTensor *_t = gguf_find_tensor(hdr, name_buf);                \
    if (!_t) {                                                              \
        fprintf(stderr, "[gguf_loader] missing tensor '%s'\n", name_buf);  \
        return TN_ERR_INVALID_WEIGHTS;                                      \
    }                                                                       \
    (type_arr)[l] = (int)_t->type;                                         \
    if (_t->type == GGUF_TYPE_Q4_K) {                                      \
        size_t _q4k_bytes = ((size_t)(n_elems) / 256) * 144;               \
        uint8_t *_heap = (uint8_t *)malloc(_q4k_bytes);                    \
        if (!_heap) return TN_ERR_OOM;                                      \
        memcpy(_heap, (const uint8_t *)_t->data, _q4k_bytes);              \
        if (store_add(store, _heap) != 0) { free(_heap); return TN_ERR_OOM; } \
        (field)[l] = (tn_i8 *)_heap;                                       \
    } else {                                                                \
        float *_f = tensor_to_f32(_t, (n_elems), store);                   \
        if (!_f) return TN_ERR_INVALID_WEIGHTS;                             \
        (field)[l] = (tn_i8 *)_f;                                          \
    }                                                                       \
} while(0)

/* Generic heap-copy loader for any Q4_K tensor → flag pointer.
 * Sets *flag_ptr = true when tensor is Q4_K (no-op for other types).
 * Used for MLA projections where all layers have consistent type. */
#define DS_LOAD_Q4K_HEAP(field, tname, n_elems, flag_ptr) do {             \
    snprintf(name_buf, sizeof(name_buf), "blk.%d." tname, l);              \
    const GGUFTensor *_t = gguf_find_tensor(hdr, name_buf);                \
    if (!_t) {                                                              \
        fprintf(stderr, "[gguf_loader] missing tensor '%s'\n", name_buf);  \
        return TN_ERR_INVALID_WEIGHTS;                                      \
    }                                                                       \
    if (_t->type == GGUF_TYPE_Q4_K) {                                      \
        size_t _q4k_bytes = ((size_t)(n_elems) / 256) * 144;               \
        uint8_t *_heap = (uint8_t *)malloc(_q4k_bytes);                    \
        if (!_heap) return TN_ERR_OOM;                                      \
        memcpy(_heap, (const uint8_t *)_t->data, _q4k_bytes);              \
        if (store_add(store, _heap) != 0) { free(_heap); return TN_ERR_OOM; } \
        (field)[l] = (tn_i8 *)_heap;                                       \
        *(flag_ptr) = true;                                                 \
    } else {                                                                \
        float *_f = tensor_to_f32(_t, (n_elems), store);                   \
        if (!_f) return TN_ERR_INVALID_WEIGHTS;                             \
        (field)[l] = (tn_i8 *)_f;                                          \
    }                                                                       \
} while(0)

/* Heap-copy loader for MLA projection weights.
 * Copies Q4_K bytes from mmap → malloc'd heap buffer.
 * Advantages vs zero-copy mmap:
 *   - Guaranteed RAM residency (never evicted under memory pressure)
 *   - 8× smaller than F32 dequant (~200 MB vs ~1.5 GB for all MLA)
 *   - Stays warm in CPU cache across tokens
 * Analogous to llama.cpp CPU_REPACK approach. */
#define DS_LOAD_MLA_PROJ_HEAP(field, tname, n_elems) do {                  \
    snprintf(name_buf, sizeof(name_buf), "blk.%d." tname, l);              \
    const GGUFTensor *_t = gguf_find_tensor(hdr, name_buf);                \
    if (!_t) {                                                              \
        fprintf(stderr, "[gguf_loader] missing tensor '%s'\n", name_buf);  \
        return TN_ERR_INVALID_WEIGHTS;                                      \
    }                                                                       \
    if (_t->type == GGUF_TYPE_Q4_K) {                                      \
        size_t _q4k_bytes = ((size_t)(n_elems) / 256) * 144;               \
        uint8_t *_heap = (uint8_t *)malloc(_q4k_bytes);                    \
        if (!_heap) return TN_ERR_OOM;                                      \
        memcpy(_heap, (const uint8_t *)_t->data, _q4k_bytes);              \
        if (store_add(store, _heap) != 0) { free(_heap); return TN_ERR_OOM; } \
        (field)[l] = (tn_i8 *)_heap;                                       \
        w->has_mla_quant = true;                                            \
    } else {                                                                \
        float *_f = tensor_to_f32(_t, (n_elems), store);                   \
        if (!_f) return TN_ERR_INVALID_WEIGHTS;                             \
        (field)[l] = (tn_i8 *)_f;                                          \
    }                                                                       \
} while(0)

    /* Allocate per-layer w2 and w13 quant type arrays */
    w->expert_w2_quant_per_layer  = (int *)calloc((size_t)nl, sizeof(int));
    w->expert_w13_quant_per_layer = (int *)calloc((size_t)nl, sizeof(int));
    if (!w->expert_w2_quant_per_layer || !w->expert_w13_quant_per_layer) return TN_ERR_OOM;
    /* Allocate per-layer type arrays for shared expert projections */
    w->shared_w1_type_per_layer = (int *)calloc((size_t)nl, sizeof(int));
    w->shared_w2_type_per_layer = (int *)calloc((size_t)nl, sizeof(int));
    w->shared_w3_type_per_layer = (int *)calloc((size_t)nl, sizeof(int));
    if (!w->shared_w1_type_per_layer || !w->shared_w2_type_per_layer ||
        !w->shared_w3_type_per_layer) return TN_ERR_OOM;

    for (int l = 0; l < nl; l++) {
        /* Attention norm */
        snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_norm.weight", l);
        w->rms_att_weight[l] = norm_to_f32(hdr, name_buf, (size_t)dim, store);
        if (!w->rms_att_weight[l]) return TN_ERR_INVALID_WEIGHTS;

        /* FFN norm */
        snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_norm.weight", l);
        w->rms_ffn_weight[l] = norm_to_f32(hdr, name_buf, (size_t)dim, store);
        if (!w->rms_ffn_weight[l]) return TN_ERR_INVALID_WEIGHTS;

        /* KV-A sub-norm (applied to kv_latent after KV-A projection) */
        snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_kv_a_norm.weight", l);
        w->rms_attn_sub_norm[l] = norm_to_f32(hdr, name_buf, (size_t)lora, store);
        if (!w->rms_attn_sub_norm[l]) return TN_ERR_INVALID_WEIGHTS;

        /* MLA attention projections — Q4K heap-copy when available, F32 fallback otherwise.
         * DS_LOAD_MLA_PROJ_HEAP sets has_mla_quant=true so MLA_MATMUL dispatches
         * to parallel_matmul_q4k (7.1× less DRAM traffic; fits in L3 per layer). */
        DS_LOAD_MLA_PROJ_HEAP(w->mla_wq,    "attn_q.weight",       (size_t)q_rows   * (size_t)dim);
        DS_LOAD_MLA_PROJ_HEAP(w->mla_wkv_a, "attn_kv_a_mqa.weight",(size_t)kva_rows * (size_t)dim);
        DS_LOAD_MLA_PROJ_HEAP(w->mla_wkv_b, "attn_kv_b.weight",    (size_t)kvb_rows * (size_t)lora);
        DS_LOAD_MLA_PROJ_HEAP(w->wo,        "attn_output.weight",   (size_t)dim      * (size_t)dim);
        w->mla_sq[l]    = 1.0f;
        w->mla_skv_a[l] = 1.0f;
        w->mla_skv_b[l] = 1.0f;
        w->so[l]        = 1.0f;

        if (l < first_dense) {
            /* Dense FFN layer */
            DS_LOAD_PROJ(w->w1, "ffn_gate.weight",
                         (size_t)hidden_dim * (size_t)dim);
            DS_LOAD_PROJ(w->w2, "ffn_down.weight",
                         (size_t)hidden_dim * (size_t)dim);
            DS_LOAD_PROJ(w->w3, "ffn_up.weight",
                         (size_t)hidden_dim * (size_t)dim);
            w->s1[l] = w->s2[l] = w->s3[l] = 1.0f;
        } else {
            /* MoE FFN layer */

            /* Router gate matrix (F32, zero-copy from mmap) */
            snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_gate_inp.weight", l);
            const GGUFTensor *tgate = gguf_find_tensor(hdr, name_buf);
            if (!tgate) {
                fprintf(stderr, "[gguf_loader] missing tensor '%s'\n", name_buf);
                return TN_ERR_INVALID_WEIGHTS;
            }
            if (tgate->type != GGUF_TYPE_F32) {
                fprintf(stderr,
                    "[gguf_loader] ffn_gate_inp.weight must be F32 (got %d)\n",
                    (int)tgate->type);
                return TN_ERR_INVALID_WEIGHTS;
            }
            w->moe_gate_w[l] = (tn_i8 *)tgate->data; /* zero-copy, F32 */
            w->moe_gate_s[l] = 1.0f;

            /* Stacked routed expert weights — kept quantized in mmap.
             * Per-expert pointers computed from the stacked tensor's base. */
            size_t expert_elems = (size_t)exp_hid * (size_t)dim;

            snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_gate_exps.weight", l);
            const GGUFTensor *tg_exps = gguf_find_tensor(hdr, name_buf);
            snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_up_exps.weight", l);
            const GGUFTensor *tu_exps = gguf_find_tensor(hdr, name_buf);
            snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_down_exps.weight", l);
            const GGUFTensor *td_exps = gguf_find_tensor(hdr, name_buf);

            if (!tg_exps || !tu_exps || !td_exps) {
                fprintf(stderr,
                    "[gguf_loader] missing expert tensors for layer %d\n", l);
                return TN_ERR_INVALID_WEIGHTS;
            }

            size_t g_stride = quant_bytes_for_elems(tg_exps->type, expert_elems);
            size_t u_stride = quant_bytes_for_elems(tu_exps->type, expert_elems);
            size_t d_stride = quant_bytes_for_elems(td_exps->type, expert_elems);

            if (g_stride == 0 || u_stride == 0 || d_stride == 0) {
                fprintf(stderr,
                    "[gguf_loader] unknown quant type for expert tensors layer %d\n", l);
                return TN_ERR_INVALID_WEIGHTS;
            }

            for (int e = 0; e < num_experts; e++) {
                w->moe_w1[l][e] = (tn_i8 *)tg_exps->data + (size_t)e * g_stride;
                w->moe_w3[l][e] = (tn_i8 *)tu_exps->data + (size_t)e * u_stride;
                w->moe_w2[l][e] = (tn_i8 *)td_exps->data + (size_t)e * d_stride;
                /* moe_s{1,2,3} unused — has_expert_quant flag drives dequant path */
                w->moe_s1[l][e] = w->moe_s2[l][e] = w->moe_s3[l][e] = 0.0f;
            }
            /* Record the actual quant type for this layer's down-proj AND gate/up experts */
            w->expert_w2_quant_per_layer[l]  = (int)td_exps->type;
            w->expert_w13_quant_per_layer[l] = (int)tg_exps->type;

            /* Shared expert weights — heap-copy as Q4_K when available.
             * Uses DS_LOAD_SHEXP_HEAP which records per-layer type for correct dispatch.
             * Different projections can have different quant types (e.g. gate/up=Q4_K, down=Q5_K). */
            DS_LOAD_SHEXP_HEAP(w->moe_shared_w1, "ffn_gate_shexp.weight",
                               (size_t)sh_hid * (size_t)dim, w->shared_w1_type_per_layer);
            DS_LOAD_SHEXP_HEAP(w->moe_shared_w2, "ffn_down_shexp.weight",
                               (size_t)sh_hid * (size_t)dim, w->shared_w2_type_per_layer);
            DS_LOAD_SHEXP_HEAP(w->moe_shared_w3, "ffn_up_shexp.weight",
                               (size_t)sh_hid * (size_t)dim, w->shared_w3_type_per_layer);
            w->moe_shared_s1[l] = w->moe_shared_s2[l] = w->moe_shared_s3[l] = 1.0f;
        }
    }

#undef DS_LOAD_PROJ

    /* Final RMS norm */
    w->rms_final_weight = norm_to_f32(hdr, "output_norm.weight", (size_t)dim, store);
    if (!w->rms_final_weight) return TN_ERR_INVALID_WEIGHTS;

    /* Set expert quant flags */
    {
        /* Determine gate/up quant type from blk.first_dense.ffn_gate_exps */
        char tn[128];
        snprintf(tn, sizeof(tn), "blk.%d.ffn_gate_exps.weight", first_dense);
        const GGUFTensor *tg = gguf_find_tensor(hdr, tn);
        snprintf(tn, sizeof(tn), "blk.%d.ffn_down_exps.weight", first_dense);
        const GGUFTensor *td = gguf_find_tensor(hdr, tn);
        w->expert_w13_quant_type = tg ? (int)tg->type : GGUF_TYPE_Q4_K;
        w->expert_w2_quant_type  = td ? (int)td->type : GGUF_TYPE_Q5_1;
        w->has_expert_quant      = true;

        /* (per-layer quant types stored in expert_w13/w2_quant_per_layer for dispatch) */
    }

    w->layers_are_ternary = false;
    w->wcls_is_ternary    = false;
    w->wcls_scale         = 1.0f;

    printf("[GGUF-DS2] Weights loaded (%d layers, %d dense + %d MoE, "
           "%d experts/layer, experts=quantized)\n",
           nl, first_dense, nl - first_dense, num_experts);
    return TN_OK;
}

/* ── weights_from_gguf() ──────────────────────────────────────────────────── */

TernaryError weights_from_gguf(TransformerWeights *w, const Config *cfg,
                                const GGUFHeader *hdr, GGUFWeightStore **store_out) {
    GGUFWeightStore *store = store_alloc();
    if (!store) return TN_ERR_OOM;
    *store_out = store;

    /* Dispatch to DeepSeek-V2 specific loader for that architecture. */
    if (strcmp(hdr->arch, "deepseek2") == 0) {
        MoEConfig mc;
        TernaryError e = moe_config_from_gguf(&mc, hdr);
        if (e != TN_OK) return e;

        /* Load token embedding (common path below) */
        const GGUFTensor *temb = gguf_find_tensor(hdr, "token_embd.weight");
        if (!temb) {
            fprintf(stderr, "[gguf_loader] missing 'token_embd.weight'\n");
            return TN_ERR_INVALID_WEIGHTS;
        }
        size_t n_emb = (size_t)cfg->vocab_size * (size_t)cfg->dim;
        float *ftmp = tensor_to_f32(temb, n_emb, store);
        if (!ftmp) return TN_ERR_INVALID_WEIGHTS;
        tn_u16 *ebuf = (tn_u16 *)malloc(n_emb * sizeof(tn_u16));
        if (!ebuf) return TN_ERR_OOM;
        for (size_t i = 0; i < n_emb; i++) {
            uint32_t bits; memcpy(&bits, &ftmp[i], 4);
            ebuf[i] = (tn_u16)(bits >> 16);
        }
        if (store_add(store, ebuf) != 0) { free(ebuf); return TN_ERR_OOM; }
        w->token_embedding_table = ebuf;

        /* Load output.weight (Q6_K) */
        const GGUFTensor *toutw = gguf_find_tensor(hdr, "output.weight");
        if (toutw) {
            float *fout = tensor_to_f32(toutw, n_emb, store);
            if (!fout) return TN_ERR_INVALID_WEIGHTS;
            tn_u16 *obuf = (tn_u16 *)malloc(n_emb * sizeof(tn_u16));
            if (!obuf) return TN_ERR_OOM;
            for (size_t i = 0; i < n_emb; i++) {
                uint32_t bits; memcpy(&bits, &fout[i], 4);
                obuf[i] = (tn_u16)(bits >> 16);
            }
            if (store_add(store, obuf) != 0) { free(obuf); return TN_ERR_OOM; }
            w->wcls = obuf;
        } else {
            w->wcls = w->token_embedding_table;
        }

        TernaryError e2 = weights_from_gguf_deepseek2(w, cfg, hdr, &mc, store);
        if (e2 != TN_OK) return e2;
        weights_build_classifier_quant(w, cfg);
        return TN_OK;
    }

    int dim        = cfg->dim;
    int hidden_dim = cfg->hidden_dim;
    int kv_dim     = config_kv_dim(cfg);
    int nl         = cfg->n_layers;
    char name_buf[128];

    /* 1. Token embedding table (vocab_size × dim).
     *
     * BF16/F16/F32 on-disk → stored as BF16 (tn_u16) for classifier (wcls),
     * embd_f32 stays NULL → embed_token uses BF16 bit-shift path.
     *
     * Quantized on-disk (Q4_K etc.) → dequant to F32 kept in embd_f32 so that
     * embed_token does a direct memcpy — exactly llama.cpp's ggml_get_rows path
     * (Q4_K → F32, no BF16 intermediate, no precision loss from truncation).
     * A separate BF16 copy is still built and kept in token_embedding_table for
     * the classifier (wcls weight-tied path). */
    {
        const GGUFTensor *t = gguf_find_tensor(hdr, "token_embd.weight");
        if (!t) {
            fprintf(stderr, "[gguf_loader] missing 'token_embd.weight'\n");
            return TN_ERR_INVALID_WEIGHTS;
        }
        size_t n = (size_t)cfg->vocab_size * (size_t)dim;
        if (t->type == GGUF_TYPE_BF16) {
            /* Zero-copy: BF16 is already uint16 layout, identical to tn_u16 */
            w->token_embedding_table = (tn_u16 *)t->data;
        } else if (t->type == GGUF_TYPE_F32) {
            /* Convert F32 → BF16: keep upper 16 bits of IEEE 754 representation */
            tn_u16 *buf = (tn_u16 *)malloc(n * sizeof(tn_u16));
            if (!buf) return TN_ERR_OOM;
            const float *src = (const float *)t->data;
            for (size_t i = 0; i < n; i++) {
                uint32_t bits; memcpy(&bits, &src[i], 4);
                buf[i] = (tn_u16)(bits >> 16);
            }
            if (store_add(store, buf) != 0) { free(buf); return TN_ERR_OOM; }
            w->token_embedding_table = buf;
        } else if (t->type == GGUF_TYPE_F16) {
            /* Dequant F16 → F32 → BF16 */
            tn_u16 *buf = (tn_u16 *)malloc(n * sizeof(tn_u16));
            if (!buf) return TN_ERR_OOM;
            const uint16_t *src = (const uint16_t *)t->data;
            for (size_t i = 0; i < n; i++) {
                float f = f16_to_f32(src[i]);
                uint32_t bits; memcpy(&bits, &f, 4);
                buf[i] = (tn_u16)(bits >> 16);
            }
            if (store_add(store, buf) != 0) { free(buf); return TN_ERR_OOM; }
            w->token_embedding_table = buf;
        } else if (t->type == GGUF_TYPE_Q4_K || t->type == GGUF_TYPE_Q8_0 ||
                   t->type == GGUF_TYPE_Q4_0 || t->type == GGUF_TYPE_Q5_0 ||
                   t->type == GGUF_TYPE_Q5_1 ||
                   t->type == GGUF_TYPE_Q5_K || t->type == GGUF_TYPE_Q6_K) {
            /* Dequantise → F32 kept as embd_f32 (exact llama.cpp path).
             * Also build BF16 copy for the classifier (wcls weight-tied). */
            float *ftmp = (float *)malloc(n * sizeof(float));
            if (!ftmp) return TN_ERR_OOM;
            switch (t->type) {
                case GGUF_TYPE_Q4_K: gguf_dequant_q4_k(ftmp, t->data, n); break;
                case GGUF_TYPE_Q8_0: gguf_dequant_q8_0(ftmp, t->data, n); break;
                case GGUF_TYPE_Q5_0: gguf_dequant_q5_0(ftmp, t->data, n); break;
                case GGUF_TYPE_Q5_1: gguf_dequant_q5_1(ftmp, t->data, n); break;
                case GGUF_TYPE_Q5_K: gguf_dequant_q5_k(ftmp, t->data, n); break;
                case GGUF_TYPE_Q6_K: gguf_dequant_q6_k(ftmp, t->data, n); break;
                default:             gguf_dequant_q4_0(ftmp, t->data, n); break;
            }
            /* Keep F32 buffer for embedding lookup — registered in store for cleanup */
            if (store_add(store, ftmp) != 0) { free(ftmp); return TN_ERR_OOM; }
            w->embd_f32 = ftmp;

            /* BF16 copy for classifier (wcls weight-tied) */
            tn_u16 *buf = (tn_u16 *)malloc(n * sizeof(tn_u16));
            if (!buf) return TN_ERR_OOM;
            for (size_t i = 0; i < n; i++) {
                uint32_t bits; memcpy(&bits, &ftmp[i], 4);
                buf[i] = (tn_u16)(bits >> 16);
            }
            if (store_add(store, buf) != 0) { free(buf); return TN_ERR_OOM; }
            w->token_embedding_table = buf;
        } else {
            fprintf(stderr,
                "[gguf_loader] unsupported embedding type %d ('%s')\n"
                "[gguf_loader] Supported embedding types: F32, F16, BF16\n",
                (int)t->type, gguf_type_name(t->type));
            return TN_ERR_INVALID_WEIGHTS;
        }
    }

    /* 2. Per-layer weights */
    /* Detect layer weight storage type from first tensor encountered.
     * All layers in a GGUF model use the same quantisation type. */
    int _layer_wtype = WEIGHT_TYPE_F32;
    for (int l = 0; l < nl; l++) {
        /* Attention norm (rms_att_weight): F32 zero-copy or dequant */
        snprintf(name_buf, sizeof(name_buf), "blk.%d.attn_norm.weight", l);
        w->rms_att_weight[l] = norm_to_f32(hdr, name_buf, (size_t)dim, store);
        if (!w->rms_att_weight[l]) return TN_ERR_INVALID_WEIGHTS;

        /* FFN norm (rms_ffn_weight) */
        snprintf(name_buf, sizeof(name_buf), "blk.%d.ffn_norm.weight", l);
        w->rms_ffn_weight[l] = norm_to_f32(hdr, name_buf, (size_t)dim, store);
        if (!w->rms_ffn_weight[l]) return TN_ERR_INVALID_WEIGHTS;

/* Helper macro: load a projection weight tensor.
 * F16/F32: zero-copy pointer into mmap'd data (no heap allocation needed).
 * Other (quantized) types: dequantize to a heap F32 buffer via tensor_to_f32().
 * Updates _layer_wtype so the model-wide dispatch flag is set after the loop. */
#define LOAD_PROJ(field, tname, n_elems) do {                               \
    snprintf(name_buf, sizeof(name_buf), "blk.%d." tname, l);              \
    const GGUFTensor *_t = gguf_find_tensor(hdr, name_buf);                \
    if (!_t) {                                                              \
        fprintf(stderr, "[gguf_loader] missing tensor '%s'\n", name_buf);  \
        return TN_ERR_INVALID_WEIGHTS;                                      \
    }                                                                       \
    if (_t->type == GGUF_TYPE_F16) {                                        \
        /* Zero-copy: F16 data lives in mmap for model lifetime */          \
        (field)[l] = (tn_i8 *)_t->data;                                    \
        _layer_wtype = WEIGHT_TYPE_F16;                                     \
    } else if (_t->type == GGUF_TYPE_F32) {                                 \
        /* Zero-copy: F32 data lives in mmap for model lifetime */          \
        (field)[l] = (tn_i8 *)_t->data;                                    \
        /* _layer_wtype stays WEIGHT_TYPE_F32 (default) */                  \
    } else {                                                                \
        float *_f = tensor_to_f32(_t, (n_elems), store);                   \
        if (!_f) return TN_ERR_INVALID_WEIGHTS;                             \
        (field)[l] = (tn_i8 *)_f;                                          \
        /* _layer_wtype stays WEIGHT_TYPE_F32 (dequanted to F32 heap) */    \
    }                                                                       \
} while(0)

        /* Attention projections */
        LOAD_PROJ(w->wq, "attn_q.weight",      (size_t)dim * (size_t)dim);
        LOAD_PROJ(w->wk, "attn_k.weight",      (size_t)kv_dim * (size_t)dim);
        LOAD_PROJ(w->wv, "attn_v.weight",      (size_t)kv_dim * (size_t)dim);
        LOAD_PROJ(w->wo, "attn_output.weight", (size_t)dim * (size_t)dim);

        /* FFN projections */
        LOAD_PROJ(w->w1, "ffn_gate.weight", (size_t)dim * (size_t)hidden_dim);
        LOAD_PROJ(w->w2, "ffn_down.weight", (size_t)hidden_dim * (size_t)dim);
        LOAD_PROJ(w->w3, "ffn_up.weight",   (size_t)dim * (size_t)hidden_dim);

#undef LOAD_PROJ
    }

    /* 3. Final RMS norm */
    w->rms_final_weight = norm_to_f32(hdr, "output_norm.weight", (size_t)dim, store);
    if (!w->rms_final_weight) return TN_ERR_INVALID_WEIGHTS;

    /* 4. Output classifier — use output.weight if present; else weight-tie. */
    {
        const GGUFTensor *t = gguf_find_tensor(hdr, "output.weight");
        if (t) {
            size_t n = (size_t)cfg->vocab_size * (size_t)dim;
            if (t->type == GGUF_TYPE_BF16) {
                w->wcls = (tn_u16 *)t->data; /* zero-copy */
            } else if (t->type == GGUF_TYPE_F32) {
                tn_u16 *buf = (tn_u16 *)malloc(n * sizeof(tn_u16));
                if (!buf) return TN_ERR_OOM;
                const float *src = (const float *)t->data;
                for (size_t i = 0; i < n; i++) {
                    uint32_t bits; memcpy(&bits, &src[i], 4);
                    buf[i] = (tn_u16)(bits >> 16);
                }
                if (store_add(store, buf) != 0) { free(buf); return TN_ERR_OOM; }
                w->wcls = buf;
            } else if (t->type == GGUF_TYPE_F16) {
                tn_u16 *buf = (tn_u16 *)malloc(n * sizeof(tn_u16));
                if (!buf) return TN_ERR_OOM;
                const uint16_t *src = (const uint16_t *)t->data;
                for (size_t i = 0; i < n; i++) {
                    float f = f16_to_f32(src[i]);
                    uint32_t bits; memcpy(&bits, &f, 4);
                    buf[i] = (tn_u16)(bits >> 16);
                }
                if (store_add(store, buf) != 0) { free(buf); return TN_ERR_OOM; }
                w->wcls = buf;
            } else {
                /* Quantized types → F32 → BF16 */
                float *ftmp = tensor_to_f32(t, n, store);
                if (!ftmp) { w->wcls = w->token_embedding_table; goto wcls_done; }
                tn_u16 *buf = (tn_u16 *)malloc(n * sizeof(tn_u16));
                if (!buf) return TN_ERR_OOM;
                for (size_t i = 0; i < n; i++) {
                    uint32_t bits; memcpy(&bits, &ftmp[i], 4);
                    buf[i] = (tn_u16)(bits >> 16);
                }
                if (store_add(store, buf) != 0) { free(buf); return TN_ERR_OOM; }
                w->wcls = buf;
            }
            wcls_done:;
        } else {
            w->wcls = w->token_embedding_table; /* weight-tied */
        }
    }

    /* 5. Set inference flags */
    w->layers_are_ternary = false;
    w->wcls_is_ternary    = false;
    w->wcls_scale         = 1.0f;
    w->layer_weight_type  = _layer_wtype;
    /* sq/sk/sv/so/s1/s2/s3 remain 0.0f (unused in non-ternary path) */
    /* rms_attn_sub_norm / rms_ffn_sub_norm remain NULL (not in standard llama) */

    /* 6. Build INT8 / INT4 quantized classifier variants */
    weights_build_classifier_quant(w, cfg);

    printf("[GGUF] Weights loaded (%d layers, arch=%s)\n", nl, hdr->arch);
    return TN_OK;
}
