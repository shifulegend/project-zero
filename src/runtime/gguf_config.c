/*
 * src/core/gguf_config.c
 *
 * Model hyperparameters read from GGUF metadata: config_from_gguf() (Config) and
 * moe_config_from_gguf() (MoEConfig). Split out of gguf_loader.c; the code is unchanged.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/moe_config.h"
#include "core/platform.h"
#include "math/rope.h"
#include "runtime/gguf_loader.h"

/* ── config_from_gguf() ───────────────────────────────────────────────────── */

TernaryError config_from_gguf(Config *cfg, const GGUFHeader *hdr) {
    memset(cfg, 0, sizeof(*cfg));

    /* Architecture prefix — "llama" for most models, "deepseek2" for DeepSeek */
    const char *arch = hdr->arch;
    char emb_key[128], blk_key[128], head_key[128], ffn_key[128];
    char kv_head_key[128], ctx_key[128], vocab_key[128], rope_key[128];
    snprintf(emb_key, sizeof(emb_key), "%s.embedding_length", arch);
    snprintf(blk_key, sizeof(blk_key), "%s.block_count", arch);
    snprintf(head_key, sizeof(head_key), "%s.attention.head_count", arch);
    snprintf(ffn_key, sizeof(ffn_key), "%s.feed_forward_length", arch);
    snprintf(kv_head_key, sizeof(kv_head_key), "%s.attention.head_count_kv", arch);
    snprintf(ctx_key, sizeof(ctx_key), "%s.context_length", arch);
    snprintf(vocab_key, sizeof(vocab_key), "%s.vocab_size", arch);
    snprintf(rope_key, sizeof(rope_key), "%s.rope.freq_base", arch);

    cfg->dim        = (int)gguf_meta_u32(hdr, emb_key, 0);
    cfg->n_layers   = (int)gguf_meta_u32(hdr, blk_key, 0);
    cfg->n_heads    = (int)gguf_meta_u32(hdr, head_key, 0);
    cfg->hidden_dim = (int)gguf_meta_u32(hdr, ffn_key, 0);

    /* Phase 22-SSM: Mamba has no attention heads or FFN (head_count = 0,
     * feed_forward_length = 0 in the file). Give the generic RunState a single
     * notional head (sized buffers stay valid; the KV cache is skipped for SSMs)
     * and size hb/hb2 by the SSM inner width so they are never zero-length. */
    if (strcmp(arch, "mamba") == 0) {
        char ik[128];
        snprintf(ik, sizeof(ik), "%s.ssm.inner_size", arch);
        if (cfg->n_heads <= 0) cfg->n_heads = 1;
        if (cfg->hidden_dim <= 0) cfg->hidden_dim = (int)gguf_meta_u32(hdr, ik, 0);
    }

    if (cfg->dim <= 0 || cfg->n_layers <= 0 || cfg->n_heads <= 0 || cfg->hidden_dim <= 0) {
        fprintf(stderr,
                "[gguf_loader] missing required metadata (embedding_length / "
                "block_count / head_count / feed_forward_length)\n"
                "[gguf_loader] Architecture in file: '%s'\n",
                arch);
        return TN_ERR_INVALID_CONFIG;
    }

    cfg->n_kv_heads = (int)gguf_meta_u32(hdr, kv_head_key, (uint32_t)cfg->n_heads);
    cfg->seq_len    = (int)gguf_meta_u32(hdr, ctx_key, 4096);
    cfg->vocab_size = (int)gguf_meta_u32(hdr, vocab_key, 0);

    /* Vocab size fallback: read from token_embd.weight tensor (dims[1] = rows) */
    if (cfg->vocab_size <= 0) {
        const GGUFTensor *embd = gguf_find_tensor(hdr, "token_embd.weight");
        if (embd && embd->n_dims >= 2) cfg->vocab_size = (int)embd->dims[1];
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

    cfg->scale_mode = 1; /* non-ternary float path */
    cfg->act_type   = 0; /* SiLU */
    /* Rotary pair layout follows the architecture (llama.cpp convention):
     * llama-family GGUFs have Q/K permuted for NORM; Qwen/Phi/Gemma/BitNet/...
     * keep NEOX. 2026-10-18: everything used NORM before — wrong for those. */
    cfg->rope_layout = rope_layout_for_arch(hdr->arch);
    /* BitNet b1.58 GGUF (arch "bitnet-b1.58", bitnet.cpp's I2_S release):
     * squared-ReLU FFN and matrix-scaled ternary projections. */
    if (strncmp(hdr->arch, "bitnet", 6) == 0) {
        cfg->act_type   = 1; /* ReLU2 */
        cfg->scale_mode = 0;
    }
    cfg->bos_token_id = (int)gguf_meta_u32(hdr, "tokenizer.ggml.bos_token_id", (uint32_t)-1);
    cfg->eos_token_id = (int)gguf_meta_u32(hdr, "tokenizer.ggml.eos_token_id", (uint32_t)-1);

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
    snprintf(scale_key, 128, "%s.rope.scaling.factor", hdr->arch);
    snprintf(scale_type_key, 128, "%s.rope.scaling.type", hdr->arch);
    snprintf(log_mul_key, 128, "%s.rope.scaling.yarn_log_multiplier", hdr->arch);
    snprintf(orig_ctx_key, 128, "%s.rope.scaling.original_context_length", hdr->arch);

    float rope_scale_factor = gguf_meta_f32(hdr, scale_key, 1.0f);
    char scale_type_buf[32];
    const char *scale_type =
        gguf_meta_str_copy(hdr, scale_type_key, scale_type_buf, sizeof(scale_type_buf)) < 0
            ? NULL
            : scale_type_buf;
    float yarn_log_mul = gguf_meta_f32(hdr, log_mul_key, 0.0f);
    int orig_ctx       = (int)gguf_meta_u32(hdr, orig_ctx_key, (uint32_t)cfg->seq_len);

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

/* ── MoE router weighting keys (2026-10-21) ─────────────────────────────────
 * Reads <arch>.expert_gating_func / expert_weights_norm / expert_weights_scale
 * with llama.cpp's per-arch defaults (see MoEConfig). An unsupported gating
 * function falls back to the default with a warning instead of silently
 * misrouting. */
static void moe_config_read_gating(MoEConfig *mc, const GGUFHeader *hdr, int default_gating,
                                   int default_norm) {
    char k[128];
    snprintf(k, sizeof(k), "%s.expert_gating_func", hdr->arch);
    int g = (int)gguf_meta_u32(hdr, k, 0);
    if (g == 0) g = default_gating;
    if (g != MOE_GATING_SOFTMAX && g != MOE_GATING_SIGMOID) {
        fprintf(stderr,
                "[gguf_loader] %s: expert_gating_func=%d not supported, "
                "using softmax\n",
                hdr->arch, g);
        g = MOE_GATING_SOFTMAX;
    }
    mc->expert_gating_func = g;

    snprintf(k, sizeof(k), "%s.expert_weights_norm", hdr->arch);
    const GGUFMeta *m       = gguf_meta_find(hdr, k);
    mc->expert_weights_norm = default_norm;
    if (m && m->val_type == GGUF_VAL_BOOL)
        mc->expert_weights_norm = m->val.boolean ? 1 : 0;
    else if (m)
        mc->expert_weights_norm = gguf_meta_u32(hdr, k, (uint32_t)default_norm) ? 1 : 0;

    snprintf(k, sizeof(k), "%s.expert_weights_scale", hdr->arch);
    mc->expert_weights_scale = gguf_meta_f32(hdr, k, 0.0f);
}

/* ── moe_config_from_gguf() ───────────────────────────────────────────────── */

TernaryError moe_config_from_gguf(MoEConfig *mc, const GGUFHeader *hdr) {
    moe_config_init_dense(mc);

    if (strcmp(hdr->arch, "mamba") == 0) {
        /* Phase 22-SSM: Mamba-1. Key names per llama.cpp (mamba.ssm.*),
         * verified against a real mamba-130m-hf GGUF header dump. */
        char k[128];
#define MK(suffix) (snprintf(k, sizeof(k), "%s." suffix, hdr->arch), k)
        mc->ssm_conv_kernel    = (int)gguf_meta_u32(hdr, MK("ssm.conv_kernel"), 0);
        mc->ssm_inner_size     = (int)gguf_meta_u32(hdr, MK("ssm.inner_size"), 0);
        mc->ssm_state_size     = (int)gguf_meta_u32(hdr, MK("ssm.state_size"), 0);
        mc->ssm_time_step_rank = (int)gguf_meta_u32(hdr, MK("ssm.time_step_rank"), 0);
#undef MK
        if (mc->ssm_conv_kernel <= 0 || mc->ssm_inner_size <= 0 || mc->ssm_state_size <= 0 ||
            mc->ssm_time_step_rank <= 0) {
            fprintf(stderr, "[gguf_loader] mamba: missing/invalid ssm.* metadata\n");
            return TN_ERR_INVALID_CONFIG;
        }
        mc->is_ssm = 1;
        return TN_OK;
    }

    if (strcmp(hdr->arch, "qwen35") == 0) {
        /* Qwen3.5/3.6 hybrid Gated-DeltaNet + Gated-Attention.
         * Not MoE (is_moe/has_mla stay 0) — a distinct 3rd attention family,
         * gated purely by has_linear_attn. Key names verified against
         * llama.cpp src/models/qwen35.cpp + a real Ternary-Bonsai-27B-Q2_0.gguf
         * header dump (see docs/ai/decision-log.md). */
        char k[128];
#define MK(suffix) (snprintf(k, sizeof(k), "%s." suffix, hdr->arch), k)
        mc->full_attention_interval = (int)gguf_meta_u32(hdr, MK("full_attention_interval"), 4);
        mc->attn_head_dim           = (int)gguf_meta_u32(hdr, MK("attention.key_length"), 0);
        mc->attn_rope_dim = (int)gguf_meta_u32(hdr, MK("rope.dimension_count"), mc->attn_head_dim);
        mc->ssm_conv_kernel    = (int)gguf_meta_u32(hdr, MK("ssm.conv_kernel"), 4);
        mc->ssm_group_count    = (int)gguf_meta_u32(hdr, MK("ssm.group_count"), 0);
        mc->ssm_inner_size     = (int)gguf_meta_u32(hdr, MK("ssm.inner_size"), 0);
        mc->ssm_state_size     = (int)gguf_meta_u32(hdr, MK("ssm.state_size"), 0);
        mc->ssm_time_step_rank = (int)gguf_meta_u32(hdr, MK("ssm.time_step_rank"), 0);
#undef MK
        if (mc->attn_head_dim <= 0 || mc->ssm_group_count <= 0 || mc->ssm_inner_size <= 0 ||
            mc->ssm_state_size <= 0 || mc->ssm_time_step_rank <= 0) {
            fprintf(stderr, "[gguf_loader] qwen35: missing/invalid ssm.* or "
                            "attention.key_length metadata\n");
            return TN_ERR_INVALID_CONFIG;
        }
        mc->has_linear_attn = 1;
        return TN_OK;
    }

    /* Dense Qwen3 whose head width is not dim/n_heads (e.g. Qwen3-0.6B:
     * key_length 128, 1024/16 = 64). The generic path derives head_dim from
     * dim/n_heads, so these run on the independent-head_dim attention of the
     * Qwen3-MoE path (QK-norm, q_rows-wide wq/wo) with a dense FFN.
     * 2026-10-19: previously silent garbage. */
    if (strcmp(hdr->arch, "qwen3") == 0) {
        uint32_t kl    = gguf_meta_u32(hdr, "qwen3.attention.key_length", 0);
        uint32_t emb   = gguf_meta_u32(hdr, "qwen3.embedding_length", 0);
        uint32_t heads = gguf_meta_u32(hdr, "qwen3.attention.head_count", 0);
        if (kl > 0 && heads > 0 && kl != emb / heads) {
            mc->attn_head_dim = (int)kl;
            mc->has_qk_norm   = 1;
            mc->is_moe        = false;
        }
        return TN_OK;
    }

    if (strcmp(hdr->arch, "qwen3moe") == 0) {
        /* Qwen3-MoE (e.g. Qwen3-30B-A3B): routed MoE, no shared experts, no
         * leading dense layers (all layers routed), plain GQA attention with
         * per-head QK-norm — not MLA, gated purely by has_qk_norm. Key names
         * follow the same expert_count/expert_used_count/
         * expert_feed_forward_length convention as deepseek2 above; NOT yet
         * confirmed against a real Qwen3-30B-A3B GGUF header dump (no such
         * file available in this environment — see docs/ai/decision-log.md
         * and docs/ai/mistakes.md for the follow-up verification needed
         * before fully trusting these against a real file). */
        char k[128];
#define MK(suffix) (snprintf(k, sizeof(k), "%s." suffix, hdr->arch), k)
        mc->num_experts         = (int)gguf_meta_u32(hdr, MK("expert_count"), 0);
        mc->num_experts_per_tok = (int)gguf_meta_u32(hdr, MK("expert_used_count"), 0);
        mc->expert_hidden_dim   = (int)gguf_meta_u32(hdr, MK("expert_feed_forward_length"), 0);
        mc->attn_head_dim       = (int)gguf_meta_u32(hdr, MK("attention.key_length"), 0);
#undef MK
        mc->first_k_dense_replace    = 0; /* all layers routed */
        mc->n_shared_experts         = 0; /* no shared experts */
        mc->shared_expert_hidden_dim = 0;
        /* llama.cpp qwen3moe.cpp: softmax gating, norm_w hardcoded true */
        moe_config_read_gating(mc, hdr, MOE_GATING_SOFTMAX, 1);

        if (mc->num_experts <= 0 || mc->num_experts_per_tok <= 0 || mc->expert_hidden_dim <= 0 ||
            mc->attn_head_dim <= 0) {
            fprintf(stderr, "[gguf_loader] qwen3moe: missing/invalid expert_count / "
                            "expert_used_count / expert_feed_forward_length / "
                            "attention.key_length metadata\n");
            return TN_ERR_INVALID_CONFIG;
        }
        mc->is_moe      = true;
        mc->has_mla     = 0;
        mc->has_qk_norm = 1;
        return TN_OK;
    }

    if (strcmp(hdr->arch, "deepseek2") != 0) return TN_OK;

    /* All keys use the model architecture prefix from the GGUF header */
    char k[128];
#define MK(suffix) (snprintf(k, sizeof(k), "%s." suffix, hdr->arch), k)

    mc->num_experts           = (int)gguf_meta_u32(hdr, MK("expert_count"), 64);
    mc->num_experts_per_tok   = (int)gguf_meta_u32(hdr, MK("expert_used_count"), 6);
    mc->expert_hidden_dim     = (int)gguf_meta_u32(hdr, MK("expert_feed_forward_length"), 1408);
    mc->first_k_dense_replace = (int)gguf_meta_u32(hdr, MK("leading_dense_block_count"), 1);
    mc->n_shared_experts      = (int)gguf_meta_u32(hdr, MK("expert_shared_count"), 2);
    mc->kv_lora_rank          = (int)gguf_meta_u32(hdr, MK("attention.kv_lora_rank"), 512);
    mc->qk_rope_head_dim      = (int)gguf_meta_u32(hdr, MK("rope.dimension_count"), 64);
    mc->v_head_dim            = (int)gguf_meta_u32(hdr, MK("attention.value_length"), 128);

    /* qk_nope = total_key_length - rope_dim */
    int key_length       = (int)gguf_meta_u32(hdr, MK("attention.key_length"), 192);
    mc->qk_nope_head_dim = key_length - mc->qk_rope_head_dim;
    if (mc->qk_nope_head_dim <= 0) mc->qk_nope_head_dim = 128; /* fallback */

#undef MK

    /* shared hidden = n_shared * expert_hidden */
    mc->shared_expert_hidden_dim = mc->n_shared_experts * mc->expert_hidden_dim;

    /* llama.cpp deepseek2.cpp: gating from metadata, softmax when absent
     * (V2 / V2.5 GGUFs carry no gating key); norm defaults to false. */
    moe_config_read_gating(mc, hdr, MOE_GATING_SOFTMAX, 0);

    mc->is_moe  = true;
    mc->has_mla = 1;
    return TN_OK;
}
