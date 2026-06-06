#!/usr/bin/env python3
"""
convert_hf_bitnet.py — HuggingFace BitNet/Qwen to Binary Converter

Phase 10.2: Converts HuggingFace models (safetensors format) into
the C engine's native binary format with 2-bit packed ternary weights.

Supports memory-efficient streaming conversion for large models (e.g., 7B+).
"""

import argparse
import json
import struct
import os
import sys
import numpy as np

try:
    import ml_dtypes
except ImportError:
    print("Warning: ml_dtypes not found. bfloat16 support might be limited.")

# Reuse the packing function from pack_ternary.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_ternary import pack_ternary_array, TN_MAGIC, TN_VERSION


def get_tensor_map(model_path):
    """Scan safetensors files and build a key -> filename map."""
    from safetensors import safe_open
    tensor_map = {}
    for fname in sorted(os.listdir(model_path)):
        if fname.endswith('.safetensors'):
            filepath = os.path.join(model_path, fname)
            with safe_open(filepath, framework="numpy") as f:
                for key in f.keys():
                    tensor_map[key] = filepath
    return tensor_map


def load_tensor(tensor_map, key):
    """Load a single small tensor (e.g., norm weights) from the correct shard."""
    from safetensors import safe_open
    if key not in tensor_map:
        return None
    filepath = tensor_map[key]
    with safe_open(filepath, framework="numpy") as f:
        return f.get_tensor(key)


def load_tensor_chunked(tensor_map, key, chunk_size=10000):
    """Generator that yields chunks of a tensor to save memory."""
    from safetensors import safe_open
    if key not in tensor_map:
        return
    
    filepath = tensor_map[key]
    with safe_open(filepath, framework="numpy") as f:
        tslice = f.get_slice(key)
        shape = tslice.get_shape()
        rows = shape[0]
        
        for start in range(0, rows, chunk_size):
            end = min(start + chunk_size, rows)
            chunk = tslice[start:end]
            yield chunk


def validate_and_quantize_ternary(tensor, scale):
    """Scale weights by absmean, round, and clip to {-1, 0, 1}."""
    tensor_f32 = tensor.astype(np.float32)
    # BitNet b1.58 quantizes by dividing by the mean absolute value
    scaled = tensor_f32 / (scale + 1e-7)
    rounded = np.round(scaled).astype(np.int8)
    return np.clip(rounded, -1, 1)


def unpack_ms_packed(packed_tensor):
    """Unpack Microsoft's BitNet row-packed uint8 format.

    HuggingFace/transformers packs a weight matrix [out, in] → packed [out//4, in] uint8.
    For packed row j at column c:
      bits[1:0] = w[j,          c]   (row j)
      bits[3:2] = w[j+row_dim,  c]   (row j + row_dim)
      bits[5:4] = w[j+2*row_dim,c]   (row j + 2*row_dim)
      bits[7:6] = w[j+3*row_dim,c]   (row j + 3*row_dim)
    where row_dim = packed_rows = out//4.
    Encoding: ternary -1→0, 0→1, 1→2 (add 1 before packing, subtract 1 on unpack).

    This matches HuggingFace transformers integrations/bitnet.py pack_weights/unpack_weights.
    """
    packed_rows, cols = packed_tensor.shape
    real_rows = packed_rows * 4

    result = np.empty((real_rows, cols), dtype=np.int8)
    for i in range(4):
        start = i * packed_rows
        end = start + packed_rows
        mask = np.uint8(3 << (2 * i))
        result[start:end] = ((packed_tensor & mask) >> (2 * i)).astype(np.int8) - 1

    return result


def process_and_write_packed(f, tensor_map, key, chunk_size=10000):
    """Process a ternary tensor in chunks and write to file."""
    if key not in tensor_map:
        print(f"    Warning: {key} not found")
        return

    # Load the FULL tensor (needed for repacking)
    full_tensor = load_tensor(tensor_map, key)
    if full_tensor is None:
        return
    
    is_already_packed = (full_tensor.dtype == np.uint8)
    
    if is_already_packed:
        # Already packed uint8 in Microsoft's row-packed format
        scale_key = key + "_scale"
        external_scale = None
        if scale_key in tensor_map:
            s_tensor = load_tensor(tensor_map, scale_key)
            if s_tensor is not None:
                # HF safetensors stores weight_scale = 1/absmean.
                # HF inference: output = W_ternary @ x * weight_scale (uses 1/absmean directly).
                # C engine dequantizes as: W_ternary @ x * stored_scale
                # So store weight_scale as-is (= 1/absmean) — same as HF convention.
                external_scale = float(np.mean(s_tensor.astype(np.float32)))
                print(f"  Found external scale for {key}: {external_scale:.4f} (1/absmean)")

        scale = external_scale if external_scale is not None else 1.0
        
        # Unpack Microsoft's format → ternary matrix → repack in our flat format
        # unpack_ms_packed handles the official 0->-1, 1->0, 2->1 mapping via (val - 1).
        print(f"  {key}: Repacking (scale={scale:.4f})...")
        ternary_matrix = unpack_ms_packed(full_tensor)
        packed_bytes = pack_ternary_array(ternary_matrix)
        f.write(packed_bytes)
        del ternary_matrix, packed_bytes
    else:
        print(f"  Processing {key} (quantizing to ternary)...")

        # 1. Compute absmean (scale) across the ENTIRE tensor
        tensor_f32 = full_tensor.astype(np.float32)
        scale = float(np.mean(np.abs(tensor_f32)))
        if scale == 0:
            scale = 1.0

        # 2. Quantize and pack
        w = validate_and_quantize_ternary(full_tensor, scale)
        f.write(pack_ternary_array(w))
        del w
        
    f.write(struct.pack('<f', scale))
    print(f"    Done. Scale (absmean): {scale:.4f}")
    del full_tensor


def process_and_write_raw(f, tensor_map, key, chunk_size=10000):
    """Write a tensor directly as float32 without packing."""
    if key not in tensor_map:
        print(f"    Warning: {key} not found")
        return
    
    print(f"  Writing {key} as float32...")
    for chunk in load_tensor_chunked(tensor_map, key, chunk_size):
        # Convert to float32 (handling bfloat16 if needed)
        f.write(chunk.astype(np.float32).tobytes())
        del chunk


def process_and_write_bf16(f, tensor_map, key, chunk_size=10000):
    """Write a tensor as native bfloat16 (2 bytes per value).

    BF16 is the native HuggingFace storage format for embeddings — no precision
    is lost compared to the source model. Saves 50% memory vs float32.
    The C engine reads these as tn_u16 and converts to float32 per-row at
    inference time (negligible cost: only dim values per token lookup).
    """
    if key not in tensor_map:
        print(f"    Warning: {key} not found")
        return

    print(f"  Writing {key} as bfloat16...")
    for chunk in load_tensor_chunked(tensor_map, key, chunk_size):
        # Reinterpret float32 bits as uint16 (upper 16 bits = bf16).
        # If already uint16/bf16, view directly; otherwise convert via float32.
        if chunk.dtype == np.uint16:
            f.write(chunk.tobytes())
        else:
            # Source may be stored as bfloat16 via ml_dtypes or as float32.
            # Go through float32 → take upper 2 bytes of each 4-byte word.
            f32 = chunk.astype(np.float32)
            u32 = f32.view(np.uint32)
            bf16 = (u32 >> 16).astype(np.uint16)
            f.write(bf16.tobytes())
        del chunk


def extract_config(model_path):
    """Read config.json from model directory."""
    config_path = os.path.join(model_path, 'config.json')
    if not os.path.exists(config_path):
        print(f"Error: config.json not found in {model_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, 'r') as f:
        cfg = json.load(f)

    config = {
        'dim': cfg.get('hidden_size', cfg.get('d_model')),
        'hidden_dim': cfg.get('intermediate_size'),
        'n_layers': cfg.get('num_hidden_layers'),
        'n_heads': cfg.get('num_attention_heads'),
        'n_kv_heads': cfg.get('num_key_value_heads', cfg.get('num_attention_heads')),
        'vocab_size': cfg.get('vocab_size'),
        'seq_len': cfg.get('max_position_embeddings', 2048),
    }
    return config


def pad_file(f, alignment=64):
    """Pad the file to the next alignment boundary."""
    current_pos = f.tell()
    padding = (alignment - (current_pos % alignment)) % alignment
    if padding > 0:
        f.write(b'\x00' * padding)


def convert(model_path, output_path):
    """Convert a model to packed binary format in a memory-efficient way."""
    print(f"Loading config...")
    config = extract_config(model_path)
    print(f"Config: {config}")

    print(f"Scanning shards...")
    tensor_map = get_tensor_map(model_path)
    if not tensor_map:
        print("Error: No tensors found", file=sys.stderr)
        sys.exit(1)

    dim = config['dim']
    hidden_dim = config['hidden_dim']
    n_layers = config['n_layers']
    vocab_size = config['vocab_size']

    print(f"\nStreaming production-ready packed model to {output_path}...")

    with open(output_path, 'wb') as f:
        # Header (41 bytes + padding to 64)
        f.write(struct.pack('<I', TN_MAGIC))
        f.write(struct.pack('<I', TN_VERSION))
        f.write(struct.pack('<iiiiiiii f i', dim, hidden_dim, n_layers,
                            config['n_heads'], config['n_kv_heads'], vocab_size,
                            config['seq_len'], 1, 500000.0, 0)) # act_type=1, rope_theta, scale_mode=0
        pad_file(f) # Finalize header at 64-byte boundary

        # Token embedding — stored as bfloat16 to save ~650 MB vs float32.
        # Zero precision loss: native HF format is bf16.
        process_and_write_bf16(f, tensor_map, 'model.embed_tokens.weight')
        pad_file(f)

        for l in range(n_layers):
            prefix = f'model.layers.{l}'
            print(f"  Layer {l}/{n_layers}...")

            # Norms (Must be raw float32)
            for norm_name in [f'{prefix}.input_layernorm.weight', f'{prefix}.post_attention_layernorm.weight']:
                process_and_write_raw(f, tensor_map, norm_name)
                pad_file(f)

            # Attention projections: Q, K, V (Packed ternary)
            for suffix in ['self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight']:
                process_and_write_packed(f, tensor_map, f'{prefix}.{suffix}')
                pad_file(f)

            # BitNet attn_sub_norm (between QKV and O projection)
            attn_sub = f'{prefix}.self_attn.attn_sub_norm.weight'
            if attn_sub in tensor_map:
                process_and_write_raw(f, tensor_map, attn_sub)
                pad_file(f)

            # O projection
            process_and_write_packed(f, tensor_map, f'{prefix}.self_attn.o_proj.weight')
            pad_file(f)

            # FFN: gate, up projections (Packed ternary)
            process_and_write_packed(f, tensor_map, f'{prefix}.mlp.gate_proj.weight')
            pad_file(f)
            process_and_write_packed(f, tensor_map, f'{prefix}.mlp.up_proj.weight')
            pad_file(f)

            # BitNet ffn_sub_norm (between gate*up and down projection)
            ffn_sub = f'{prefix}.mlp.ffn_sub_norm.weight'
            if ffn_sub in tensor_map:
                process_and_write_raw(f, tensor_map, ffn_sub)
                pad_file(f)

            # Down projection
            process_and_write_packed(f, tensor_map, f'{prefix}.mlp.down_proj.weight')
            pad_file(f)

        # Final norm (Must be raw float32)
        process_and_write_raw(f, tensor_map, 'model.norm.weight')
        pad_file(f)

        # Output classifier (Only if explicitly in shards, otherwise engine ties to embeddings)
        if 'lm_head.weight' in tensor_map:
            process_and_write_packed(f, tensor_map, 'lm_head.weight')
            pad_file(f)

        # Add fast global length footprint and metadata checksum
        written_bytes = f.tell()
        # Create a simple checksum of the metadata config values
        meta_checksum = (dim ^ config['hidden_dim'] ^ n_layers ^ config['n_heads'] ^ vocab_size) & 0xFFFFFFFF
        f.write(struct.pack('<QI', written_bytes, meta_checksum))

    print(f"\nDone! Output: {output_path} ({os.path.getsize(output_path):,} bytes)")


def main():
    parser = argparse.ArgumentParser(description='Memory-efficient model converter')
    parser.add_argument('--model', required=True, help='Model directory')
    parser.add_argument('--output', required=True, help='Output .bin file')
    args = parser.parse_args()
    convert(args.model, args.output)


if __name__ == '__main__':
    main()
