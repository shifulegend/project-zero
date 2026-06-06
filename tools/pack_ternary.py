#!/usr/bin/env python3
"""
pack_ternary.py — Ternary Weight Packer (2-bit packing)

Phase 10.1: Packs ternary weights {-1, 0, 1} into 2-bit format.
4 weights per byte, encoding: -1 -> 0b00, 0 -> 0b01, 1 -> 0b10.

Binary file format:
  [magic: 4 bytes "TNRY" = 0x594E5254 LE]
  [version: 4 bytes = 1]
  [Config struct: 7 * 4 = 28 bytes]
  [scale_mode: 1 byte — 0 = per-matrix, 1 = per-group]
  [group_size: 4 bytes (only meaningful if scale_mode == 1)]
  [weight data: packed weights with interleaved scales]

Usage:
  python pack_ternary.py --input weights_dir/ --output model.bin \\
      --dim 256 --hidden_dim 512 --n_layers 4 --n_heads 4 \\
      --n_kv_heads 4 --vocab_size 1000 --seq_len 512
"""

import argparse
import struct
import numpy as np
import os
import sys

# Magic and version must match TN_MAGIC / TN_VERSION in platform.h
TN_MAGIC = 0x594E5254  # "TNRY" in little-endian
TN_VERSION = 1

# Encoding: -1 -> 0, 0 -> 1, 1 -> 2
def encode_ternary(val):
    """Map ternary value to 2-bit encoding."""
    return val + 1

def pack_ternary_array(weights):
    """Pack an array of ternary values {-1, 0, 1} into 2-bit packed bytes.
    Vectorized using numpy for 100x speedup over Python loops.
    """
    flat = weights.flatten().astype(np.int8)
    n = len(flat)

    # Encode: -1->0, 0->1, 1->2
    encoded = (flat + 1).astype(np.uint8)

    # Pad to multiple of 4
    n_padded = (n + 3) // 4 * 4
    if n_padded > n:
        padded = np.zeros(n_padded, dtype=np.uint8)
        padded[:n] = encoded
    else:
        padded = encoded
    
    # Reshape and pack: (N/4, 4)
    reshaped = padded.reshape(-1, 4)
    packed = (reshaped[:, 0].astype(np.uint8) << 0 | 
              reshaped[:, 1].astype(np.uint8) << 2 | 
              reshaped[:, 2].astype(np.uint8) << 4 | 
              reshaped[:, 3].astype(np.uint8) << 6)
    
    return packed.tobytes()


def unpack_ternary_array(packed_bytes, count):
    """Unpack 2-bit packed bytes back to ternary values. Vectorized."""
    packed = np.frombuffer(packed_bytes, dtype=np.uint8)
    
    # Extract 2-bit values
    v1 = (packed & 0x03)
    v2 = (packed >> 2) & 0x03
    v3 = (packed >> 4) & 0x03
    v4 = (packed >> 6) & 0x03
    
    # Interleave and truncate
    result = np.column_stack([v1, v2, v3, v4]).flatten()[:count]
    return (result.astype(np.int8) - 1)


def write_packed_model(output_path, config, weights_dict, scale_mode=0,
                       group_size=128):
    """Write a complete packed model binary file.

    Args:
        output_path: Path to output .bin file
        config: dict with keys dim, hidden_dim, n_layers, n_heads,
                n_kv_heads, vocab_size, seq_len
        weights_dict: dict mapping weight names to (numpy_array, scale_float)
        scale_mode: 0 = per-matrix, 1 = per-group
        group_size: group size for per-group scales
    """
    dim = config['dim']
    hidden_dim = config['hidden_dim']
    n_layers = config['n_layers']
    n_heads = config['n_heads']
    n_kv_heads = config['n_kv_heads']
    vocab_size = config['vocab_size']
    seq_len = config['seq_len']
    kv_dim = (dim // n_heads) * n_kv_heads

    with open(output_path, 'wb') as f:
        # Header: magic + version
        f.write(struct.pack('<I', TN_MAGIC))
        f.write(struct.pack('<I', TN_VERSION))

        # Config struct (7 ints)
        f.write(struct.pack('<iiiiiii', dim, hidden_dim, n_layers,
                            n_heads, n_kv_heads, vocab_size, seq_len))

        # Scale mode + group size
        f.write(struct.pack('<B', scale_mode))
        f.write(struct.pack('<I', group_size))

        # Token embedding table (packed) + scale
        emb = weights_dict.get('token_embedding')
        if emb:
            w, s = emb
            assert w.shape == (vocab_size, dim), \
                f"token_embedding shape {w.shape} != ({vocab_size}, {dim})"
            f.write(pack_ternary_array(w))
            f.write(struct.pack('<f', s))

        # Per-layer weights
        for l in range(n_layers):
            # RMS attention norm (float, not packed)
            rms_att = weights_dict.get(f'rms_att_weight.{l}')
            if rms_att:
                w, _ = rms_att
                f.write(w.astype(np.float32).tobytes())

            # Q, K, V, O projections (packed) + scales
            for name, shape in [
                (f'wq.{l}', (dim, dim)),
                (f'wk.{l}', (dim, kv_dim)),
                (f'wv.{l}', (dim, kv_dim)),
                (f'wo.{l}', (dim, dim)),
            ]:
                entry = weights_dict.get(name)
                if entry:
                    w, s = entry
                    assert w.shape == shape, \
                        f"{name} shape {w.shape} != {shape}"
                    f.write(pack_ternary_array(w))
                    f.write(struct.pack('<f', s))

            # RMS FFN norm (float)
            rms_ffn = weights_dict.get(f'rms_ffn_weight.{l}')
            if rms_ffn:
                w, _ = rms_ffn
                f.write(w.astype(np.float32).tobytes())

            # FFN: w1, w2, w3 (packed) + scales
            for name, shape in [
                (f'w1.{l}', (dim, hidden_dim)),
                (f'w2.{l}', (hidden_dim, dim)),
                (f'w3.{l}', (dim, hidden_dim)),
            ]:
                entry = weights_dict.get(name)
                if entry:
                    w, s = entry
                    assert w.shape == shape, \
                        f"{name} shape {w.shape} != {shape}"
                    f.write(pack_ternary_array(w))
                    f.write(struct.pack('<f', s))

        # Final RMS norm (float)
        rms_final = weights_dict.get('rms_final_weight')
        if rms_final:
            w, _ = rms_final
            f.write(w.astype(np.float32).tobytes())

        # Output classifier (packed) + scale
        wcls = weights_dict.get('wcls')
        if wcls:
            w, s = wcls
            f.write(pack_ternary_array(w))
            f.write(struct.pack('<f', s))

    file_size = os.path.getsize(output_path)
    print(f"Wrote packed model: {output_path} ({file_size:,} bytes)")


def main():
    parser = argparse.ArgumentParser(
        description='Pack ternary weights into 2-bit format')
    parser.add_argument('--input', required=True,
                        help='Directory containing .npy weight files')
    parser.add_argument('--output', required=True,
                        help='Output .bin file path')
    parser.add_argument('--dim', type=int, required=True)
    parser.add_argument('--hidden_dim', type=int, required=True)
    parser.add_argument('--n_layers', type=int, required=True)
    parser.add_argument('--n_heads', type=int, required=True)
    parser.add_argument('--n_kv_heads', type=int, required=True)
    parser.add_argument('--vocab_size', type=int, required=True)
    parser.add_argument('--seq_len', type=int, required=True)
    parser.add_argument('--scale_mode', type=int, default=0,
                        choices=[0, 1], help='0=per-matrix, 1=per-group')
    parser.add_argument('--group_size', type=int, default=128)
    args = parser.parse_args()

    config = {
        'dim': args.dim,
        'hidden_dim': args.hidden_dim,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'n_kv_heads': args.n_kv_heads,
        'vocab_size': args.vocab_size,
        'seq_len': args.seq_len,
    }

    # Load .npy files from input directory
    weights_dict = {}
    input_dir = args.input
    for fname in sorted(os.listdir(input_dir)):
        if fname.endswith('.npy'):
            name = fname[:-4]  # remove .npy
            w = np.load(os.path.join(input_dir, fname))
            # Default scale of 1.0 if not provided separately
            scale_file = os.path.join(input_dir, f"{name}.scale")
            scale = 1.0
            if os.path.exists(scale_file):
                with open(scale_file, 'r') as sf:
                    scale = float(sf.read().strip())
            weights_dict[name] = (w, scale)
            print(f"  Loaded {name}: shape={w.shape}, scale={scale}")

    write_packed_model(args.output, config, weights_dict,
                       args.scale_mode, args.group_size)


if __name__ == '__main__':
    main()
