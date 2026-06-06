"""
convert_hf.py — HuggingFace → project-zero .bin converter
Supports:
  • Dense FP32 models (original)
  • Ternary-quantized MoE models with --moe flag (DeepSeek-V2-Lite, Mixtral)

Binary format (ternary MoE, scale_mode=2):
  [0..7]    Magic (0x594E5254) + Version (1)
  [8..47]   Config struct (40 bytes, 10 fields)
  [48..63]  Padding
  [64..95]  MoE config (32 bytes)
  [96..127] Padding
  [128+]    Weight data:
              embed_tokens (bf16) ALIGN64
              Per layer:
                rms_att_weight (fp32) ALIGN64
                rms_ffn_weight (fp32) ALIGN64
                wq + sq, wk + sk, wv + sv   (ternary + scale each)
                wo + so
                [Dense layers only: w1+s1, w3+s3, w2+s2]
              rms_final_weight (fp32) ALIGN64
              [Per MoE layer]:
                gate_w + gate_s
                [Per expert]: w1+s1, w3+s3, w2+s2
                [Shared expert]: sw1+ss1, sw3+ss3, sw2+ss2
"""
import os, sys, struct, json, math
import numpy as np
import ml_dtypes  # registers bfloat16 with numpy
from safetensors import safe_open
from argparse import ArgumentParser

TN_MAGIC   = 0x594E5254
TN_VERSION = 1

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def align64(f):
    pos = f.tell()
    rem = pos % 64
    if rem:
        f.write(b'\x00' * (64 - rem))

def packed_bytes(n):
    """Bytes needed to store n ternary values at 4 per byte (2 bits each)."""
    return (n + 3) // 4

def quantize_ternary(w: np.ndarray):
    """
    BitNet-style ternary quantization.
    scale = mean(|w|)  (L1 norm scale)
    q = round(w / scale), clipped to {-1, 0, +1}
    Returns (scale: float32, packed: bytes)
    """
    w = w.astype(np.float32).flatten()
    scale = np.mean(np.abs(w))
    if scale < 1e-9:
        scale = 1.0
    q = np.clip(np.round(w / scale), -1, 1).astype(np.int8)

    # Pack 4 values per byte: -1→00, 0→01, 1→10  (LSB first)
    enc = (q + 1).astype(np.uint8)   # {-1,0,+1} → {0,1,2}
    nb = packed_bytes(len(enc))
    # Vectorized packing — pad to multiple of 4, reshape, bitshift and OR
    padded = np.zeros(nb * 4, dtype=np.uint8)
    padded[:len(enc)] = enc
    g = padded.reshape(-1, 4)
    buf = (g[:, 0] | (g[:, 1] << 2) | (g[:, 2] << 4) | (g[:, 3] << 6)).tobytes()
    return np.float32(scale), buf

def write_ternary_block(f, w: np.ndarray):
    """Write packed ternary weight block + float32 scale, each 64-byte aligned."""
    scale, packed = quantize_ternary(w)
    f.write(packed)
    align64(f)
    f.write(struct.pack('<f', float(scale)))
    align64(f)

def write_fp32_block(f, w: np.ndarray):
    """Write float32 weight block, 64-byte aligned."""
    f.write(w.astype(np.float32).tobytes())
    align64(f)

def write_bf16_block(f, w: np.ndarray):
    """Write bfloat16 weight block (native HF format), 64-byte aligned."""
    # bf16 = upper 2 bytes of float32
    f32 = w.astype(np.float32)
    # Reinterpret as uint32 and take upper 16 bits
    u32 = f32.view(np.uint32)
    bf16 = (u32 >> 16).astype(np.uint16)
    f.write(bf16.tobytes())
    align64(f)

# ──────────────────────────────────────────────────────────────────────────────
# Weight loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_weight_map(model_dir):
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    weight_map = {}
    if os.path.exists(index_path):
        with open(index_path) as fh:
            weight_map = json.load(fh)["weight_map"]
    else:
        for fname in os.listdir(model_dir):
            if fname.endswith(".safetensors"):
                with safe_open(os.path.join(model_dir, fname), framework="numpy", device="cpu") as st:
                    for k in st.keys():
                        weight_map[k] = fname
    return weight_map

def get_tensor(model_dir, weight_map, name):
    shard = weight_map.get(name)
    if not shard:
        return None
    path = os.path.join(model_dir, shard)
    with safe_open(path, framework="numpy", device="cpu") as st:
        return st.get_tensor(name).astype(np.float32)

# ──────────────────────────────────────────────────────────────────────────────
# MLA → standard attention expansion
#
# DeepSeek-V2 uses Multi-head Latent Attention (MLA):
#   compressed_kv  = kv_a_proj_with_mqa[:kv_lora_rank, :] @ x   → (512,)
#   k_nope         = kv_b_proj[:n_heads*qk_nope_head_dim, :] @ norm(compressed_kv)
#   v              = kv_b_proj[n_heads*qk_nope_head_dim:, :] @ norm(compressed_kv)
#   k_rope         = kv_a_proj_with_mqa[kv_lora_rank:, :] @ x   → (qk_rope_head_dim,)
#
# For our standard GQA engine (uniform head_dim = dim/n_heads = 128):
#   We APPROXIMATE by expanding the nope K projection offline:
#     k_proj = kv_b_proj[:n_heads*qk_nope_head_dim, :] @ kv_a_proj_with_mqa[:kv_lora_rank, :]
#            → (n_heads*qk_nope_head_dim, dim) = (2048, 2048)
#     v_proj = kv_b_proj[n_heads*qk_nope_head_dim:, :] @ kv_a_proj_with_mqa[:kv_lora_rank, :]
#            → (n_heads*v_head_dim, dim) = (2048, 2048)
#
# The q_proj is (n_heads * (qk_nope_head_dim + qk_rope_head_dim), dim) = (3072, 2048).
# We truncate to (n_heads * qk_nope_head_dim, dim) = (2048, 2048) — takes only nope dims.
#
# NOTE: This approximation produces incorrect outputs (RoPE / decoupled-attention
# information is lost), but preserves the correct FLOP count for benchmarking.
# ──────────────────────────────────────────────────────────────────────────────

def expand_mla_attention(model_dir, weight_map, layer, cfg_hf):
    """DEPRECATED — kept for reference only.
    This function pre-multiplied kv_b × kv_a to produce standard wk/wv matrices,
    which loses k_rope (shared positional encoding) and truncates q_rope heads.
    Result: repetitive garbage output ('blueprint blueprint...').
    Use write_mla_attention() instead (Phase 17.9).
    """
    raise RuntimeError(
        "expand_mla_attention() is deprecated. Use write_mla_attention() for correct MLA output.")


def write_mla_attention(f, model_dir, weight_map, layer, cfg_hf):
    """Phase 17.9 — Write raw MLA projection matrices (no pre-multiplication).

    Writes three ternary-quantised blocks per layer:
      [mla_wq]    full Q projection (n_heads*(qk_nope+qk_rope) × dim)
      [mla_wkv_a] KV compress projection ((kv_lora_rank+qk_rope) × dim)
      [mla_wkv_b] KV expand projection (n_kv_heads*(qk_nope+v_head) × kv_lora_rank)

    These are read back by moe_weights_map() when has_mla=1.
    The output projection (wo) is written by the caller after this function returns.
    """
    pfx = f"model.layers.{layer}.self_attn"

    q_full = get_tensor(model_dir, weight_map, f"{pfx}.q_proj.weight")
    kv_a   = get_tensor(model_dir, weight_map, f"{pfx}.kv_a_proj_with_mqa.weight")
    kv_b   = get_tensor(model_dir, weight_map, f"{pfx}.kv_b_proj.weight")

    write_ternary_block(f, q_full)   # mla_wq  + mla_sq
    write_ternary_block(f, kv_a)     # mla_wkv_a + mla_skv_a
    write_ternary_block(f, kv_b)     # mla_wkv_b + mla_skv_b

# ──────────────────────────────────────────────────────────────────────────────
# Dense model conversion (FP32, original behaviour)
# ──────────────────────────────────────────────────────────────────────────────

def convert_dense(args):
    with open(os.path.join(args.model, "config.json")) as fh:
        cfg = json.load(fh)

    dim        = cfg.get("hidden_size", cfg.get("dim"))
    hidden_dim = cfg.get("intermediate_size", cfg.get("hidden_dim"))
    n_layers   = cfg.get("num_hidden_layers", cfg.get("n_layers"))
    n_heads    = cfg.get("num_attention_heads", cfg.get("n_heads"))
    n_kv_heads = cfg.get("num_key_value_heads", n_heads)
    vocab_size = cfg.get("vocab_size")
    seq_len    = cfg.get("max_position_embeddings", 2048)
    rope_theta = cfg.get("rope_theta", 10000.0)
    act_type   = 1 if cfg.get("hidden_act") == "relu2" else 0

    print(f"Dense: {dim=}, {hidden_dim=}, {n_layers=}, {n_heads=}, {vocab_size=}")

    weight_map = build_weight_map(args.model)
    def get(name):
        return get_tensor(args.model, weight_map, name)

    with open(args.output, "wb") as f:
        f.write(struct.pack("<I", TN_MAGIC))
        f.write(struct.pack("<I", TN_VERSION))
        f.write(struct.pack("<iiiiiiii f i",
                            dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                            vocab_size, seq_len, act_type, rope_theta, 1))  # scale_mode=1
        align64(f)  # pad to 64

        def write_fp32(name):
            arr = get(name)
            if arr is None:
                if name != "lm_head.weight":
                    print(f"  Warning: {name} not found")
                return
            f.write(arr.astype(np.float32).tobytes())
            align64(f)

        write_fp32("model.embed_tokens.weight")
        for l in range(n_layers):
            print(f"  Layer {l}/{n_layers}...", end="\r")
            pfx = f"model.layers.{l}"
            write_fp32(f"{pfx}.input_layernorm.weight")
            write_fp32(f"{pfx}.post_attention_layernorm.weight")
            write_fp32(f"{pfx}.self_attn.q_proj.weight")
            write_fp32(f"{pfx}.self_attn.k_proj.weight")
            write_fp32(f"{pfx}.self_attn.v_proj.weight")
            write_fp32(f"{pfx}.self_attn.o_proj.weight")
            write_fp32(f"{pfx}.mlp.gate_proj.weight")
            write_fp32(f"{pfx}.mlp.down_proj.weight")
            write_fp32(f"{pfx}.mlp.up_proj.weight")
        write_fp32("model.norm.weight")

    print(f"\nDone! Output: {args.output}")

# ──────────────────────────────────────────────────────────────────────────────
# MoE conversion — BitNet-style ternary packing
#
# IMPORTANT — per CPU_LLM_TERNARY_ENGINE.md §"Category 3: Standard Models":
#   This converter is intended for models whose weights are ALREADY ternary
#   (-1, 0, +1) or very close to it (e.g. BitNet b1.58 variants).
#   Standard float models (DeepSeek-V2, Llama, Mistral, etc.) must NOT be
#   forced through ternary quantization — the result is severe intelligence
#   loss ("brain damage").  Use GGUF Q4_K_S or Q8_0 format for those:
#     ./llama-quantize model.safetensors model.gguf Q4_K_S
#   The engine's GGUF loader (gguf_loader.c) handles Q4_K and Q8_0 natively.
# ──────────────────────────────────────────────────────────────────────────────

def convert_moe(args):
    with open(os.path.join(args.model, "config.json")) as fh:
        cfg_hf = json.load(fh)

    # ── Config fields ──
    dim               = cfg_hf["hidden_size"]
    hidden_dim_dense  = cfg_hf["intermediate_size"]   # layer-0 dense FFN
    n_layers          = cfg_hf["num_hidden_layers"]
    n_heads           = cfg_hf["num_attention_heads"]
    n_kv_heads        = cfg_hf.get("num_key_value_heads", n_heads)
    vocab_size        = cfg_hf["vocab_size"]
    seq_len           = min(cfg_hf.get("max_position_embeddings", 4096), 4096)
    rope_theta        = cfg_hf.get("rope_theta", 10000.0)
    act_type          = 1 if cfg_hf.get("hidden_act") == "relu2" else 0

    # ── MoE config fields ──
    n_routed_experts      = cfg_hf.get("n_routed_experts", cfg_hf.get("num_experts", 64))
    top_k                 = cfg_hf.get("num_experts_per_tok", 6)
    expert_hidden_dim     = cfg_hf.get("moe_intermediate_size", cfg_hf.get("ffn_dim", 1408))
    first_k_dense         = cfg_hf.get("first_k_dense_replace", 0)
    n_shared_experts      = cfg_hf.get("n_shared_experts", 0)
    shared_expert_hdim    = n_shared_experts * expert_hidden_dim  # fold into 1 virtual expert

    # ── MLA config fields (DeepSeek-style) ──
    kv_lora_rank      = cfg_hf.get("kv_lora_rank", 0)
    qk_nope_head_dim  = cfg_hf.get("qk_nope_head_dim", 0)
    qk_rope_head_dim  = cfg_hf.get("qk_rope_head_dim", 0)
    v_head_dim_mla    = cfg_hf.get("v_head_dim", 0)
    has_mla           = 1 if kv_lora_rank > 0 else 0

    print(f"MoE model: dim={dim}, layers={n_layers}, experts={n_routed_experts}, "
          f"top_k={top_k}, expert_hdim={expert_hidden_dim}, "
          f"first_k_dense={first_k_dense}, shared={n_shared_experts}(hdim={shared_expert_hdim})")
    if has_mla:
        print(f"MLA: kv_lora_rank={kv_lora_rank}, qk_nope={qk_nope_head_dim}, "
              f"qk_rope={qk_rope_head_dim}, v_head={v_head_dim_mla}")

    weight_map = build_weight_map(args.model)
    def get(name):
        return get_tensor(args.model, weight_map, name)

    with open(args.output, "wb") as f:
        # ── Header (64 bytes total) ──
        # scale_mode=2: ternary + MoE, no sub-norms
        f.write(struct.pack("<I", TN_MAGIC))
        f.write(struct.pack("<I", TN_VERSION))
        f.write(struct.pack("<iiiiiiii f i",
                            dim, hidden_dim_dense, n_layers, n_heads, n_kv_heads,
                            vocab_size, seq_len, act_type, rope_theta, 2))  # scale_mode=2
        align64(f)   # offset now = 64

        # ── MoE config header (48 bytes at offset 64) ──
        # Layout: 7 existing int32 fields (28 bytes) + 5 MLA int32 fields (20 bytes) = 48 bytes.
        # Old readers see only the first 32 bytes (7 fields + 4-byte padding they expected).
        # New readers get has_mla + 4 MLA dims from bytes 28–47.
        f.write(struct.pack("<iiiii",
                            n_routed_experts, top_k, expert_hidden_dim,
                            first_k_dense, 1))           # is_moe=1
        f.write(struct.pack("<ii",
                            1 if n_shared_experts > 0 else 0,   # n_shared_experts (folded)
                            shared_expert_hdim))          # shared_expert_hidden_dim
        # MLA extension: bytes 28-47 (5 × int32)
        f.write(struct.pack("<iiiii",
                            has_mla, kv_lora_rank,
                            qk_nope_head_dim, qk_rope_head_dim,
                            v_head_dim_mla))              # 20 bytes → total 48 bytes
        align64(f)   # pad to 128

        # ── 1. Embedding table (bf16) ──
        print("Writing embed_tokens...")
        emb = get("model.embed_tokens.weight")
        write_bf16_block(f, emb)

        # ── 2. Per-layer regular section ──
        for l in range(n_layers):
            pfx = f"model.layers.{l}"
            is_moe = (l >= first_k_dense)
            print(f"  Layer {l}/{n_layers} ({'MoE' if is_moe else 'dense'})...", end="\r")

            # Norms
            rms_att = get(f"{pfx}.input_layernorm.weight")
            rms_ffn = get(f"{pfx}.post_attention_layernorm.weight")
            write_fp32_block(f, rms_att)
            write_fp32_block(f, rms_ffn)

            # Attention weights: use write_mla_attention() for MLA models (Phase 17.9),
            # or standard q/k/v for non-MLA MoE models (e.g. Mixtral).
            if has_mla:
                write_mla_attention(f, args.model, weight_map, l, cfg_hf)
            else:
                # Non-MLA MoE (Mixtral style): standard wq/wk/wv
                pfx_attn = f"model.layers.{l}.self_attn"
                q_proj = get_tensor(args.model, weight_map, f"{pfx_attn}.q_proj.weight")
                k_proj = get_tensor(args.model, weight_map, f"{pfx_attn}.k_proj.weight")
                v_proj = get_tensor(args.model, weight_map, f"{pfx_attn}.v_proj.weight")
                write_ternary_block(f, q_proj)
                write_ternary_block(f, k_proj)
                write_ternary_block(f, v_proj)
            wo = get_tensor(args.model, weight_map, f"model.layers.{l}.self_attn.o_proj.weight")
            # No attn_sub_norm for DeepSeek (scale_mode=2 skips it)
            write_ternary_block(f, wo)      # wo + so

            # Dense FFN only for non-MoE layers
            if not is_moe:
                w1 = get(f"{pfx}.mlp.gate_proj.weight")
                w3 = get(f"{pfx}.mlp.up_proj.weight")
                w2 = get(f"{pfx}.mlp.down_proj.weight")
                write_ternary_block(f, w1)
                write_ternary_block(f, w3)
                # No ffn_sub_norm (scale_mode=2)
                write_ternary_block(f, w2)

        print()

        # ── 3. Final norm ──
        print("Writing final norm...")
        rms_final = get("model.norm.weight")
        write_fp32_block(f, rms_final)

        # (Classifier is weight-tied to embedding table in the engine — not written)

        # ── 4. MoE expert weights (per MoE layer) ──
        for l in range(first_k_dense, n_layers):
            pfx = f"model.layers.{l}"
            print(f"  MoE layer {l}/{n_layers} experts...", end="\r")

            # Gate weight: [n_routed_experts × dim]
            gate_w = get(f"{pfx}.mlp.gate.weight")   # (64, 2048)
            write_ternary_block(f, gate_w)

            # Per-expert weights
            for e in range(n_routed_experts):
                epfx = f"{pfx}.mlp.experts.{e}"
                w1 = get(f"{epfx}.gate_proj.weight")   # (1408, 2048)
                w3 = get(f"{epfx}.up_proj.weight")     # (1408, 2048)
                w2 = get(f"{epfx}.down_proj.weight")   # (2048, 1408)
                write_ternary_block(f, w1)
                write_ternary_block(f, w3)
                write_ternary_block(f, w2)

            # Shared expert (folded: n_shared × expert_hidden_dim → 1 virtual expert)
            if n_shared_experts > 0:
                spfx = f"{pfx}.mlp.shared_experts"
                sw1 = get(f"{spfx}.gate_proj.weight")   # (2816, 2048)
                sw3 = get(f"{spfx}.up_proj.weight")     # (2816, 2048)
                sw2 = get(f"{spfx}.down_proj.weight")   # (2048, 2816)
                write_ternary_block(f, sw1)
                write_ternary_block(f, sw3)
                write_ternary_block(f, sw2)

        print(f"\nDone! Output: {args.output}")
        size_gb = os.path.getsize(args.output) / 1e9
        print(f"File size: {size_gb:.2f} GB")

# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = ArgumentParser(description="Convert HuggingFace model to project-zero .bin format")
    parser.add_argument("--model",  type=str, required=True, help="HF model directory")
    parser.add_argument("--output", type=str, required=True, help="Output .bin file path")
    parser.add_argument("--moe",    action="store_true",
                        help="Convert as MoE model (DeepSeek-V2-Lite, Mixtral) with ternary quantization")
    args = parser.parse_args()

    if args.moe:
        convert_moe(args)
    else:
        convert_dense(args)

if __name__ == "__main__":
    main()
