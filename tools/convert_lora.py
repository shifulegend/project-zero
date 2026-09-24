"""
convert_lora.py — HuggingFace PEFT LoRA adapter -> project-zero .lora.bin converter

Reads a standard PEFT LoRA adapter directory (adapter_config.json +
adapter_model.safetensors) and writes the .lora.bin format consumed by
lora_load() (include/core/lora.h). No reshape/transpose is needed anywhere
in this converter: PEFT's own lora_A.weight ([rank, in_features]) and
lora_B.weight ([out_features, rank]) tensor shapes already match the
row-major-by-output-row layout lora_apply() (src/math/lora_matmul.c) expects
— see lora.h's format doc for the full byte layout.

Usage:
    python tools/convert_lora.py --adapter /path/to/hf/adapter_dir \\
        --output my_adapter.lora.bin \\
        --dim 576 --kv-dim 192 --hidden-dim 1536 --n-layers 30

--dim/--kv-dim/--hidden-dim/--n-layers must match the *base model's* Config
exactly (they're printed in project-zero's own "Model Configuration:"
startup banner for any GGUF model — kv_dim is printed there directly) —
lora_load() rejects a mismatch rather than silently producing garbage
output (see lora_load()'s header-vs-base-model shape check in
src/core/lora_load.c). --kv-dim matters only when the adapter targets
k_proj/v_proj on a GQA model (n_kv_heads < n_heads, e.g. SmolLM2): K/V
project dim -> kv_dim, not dim -> dim, a real bug found and fixed during
Phase 19 development against an actual downloaded SmolLM2 LoRA adapter
(see docs/ai/mistakes.md).
"""
import os
import re
import json
import struct
from argparse import ArgumentParser

import numpy as np
from safetensors import safe_open

LORA_BIN_MAGIC = 0x41524F4C  # "LORA" LE, matches include/core/lora.h
LORA_BIN_VERSION = 1

# Order MUST match include/core/lora.h's LoRaTargetBit enum exactly — the
# target_mask bit position and the on-disk block order both key off this.
TARGETS = [
    ("q_proj",    0),  # LORA_TARGET_Q
    ("k_proj",    1),  # LORA_TARGET_K
    ("v_proj",    2),  # LORA_TARGET_V
    ("o_proj",    3),  # LORA_TARGET_O
    ("gate_proj", 4),  # LORA_TARGET_GATE
    ("up_proj",   5),  # LORA_TARGET_UP
    ("down_proj", 6),  # LORA_TARGET_DOWN
]


def find_adapter_weights_file(adapter_dir):
    for name in ("adapter_model.safetensors",):
        path = os.path.join(adapter_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"no adapter_model.safetensors found in {adapter_dir} "
        "(only the safetensors PEFT format is supported, not the legacy "
        "adapter_model.bin pickle format)"
    )


def find_lora_key(keys, layer, hf_module_name, matrix):
    """Locates the lora_A/lora_B tensor key for one (layer, module), tolerant
    of PEFT's varying key prefixes (base_model.model.model.layers.N... vs.
    base_model.model.layers.N..., different attention class names, etc.) —
    matches on the stable ".layers.{layer}." + "{module}.lora_{matrix}.weight"
    suffix shared by every PEFT export, rather than assuming one exact
    prefix. Raises if zero or more than one key matches (ambiguous match is
    a converter bug, not something to silently guess through)."""
    pattern = re.compile(
        rf"\.layers\.{layer}\.[a-zA-Z0-9_.]*{re.escape(hf_module_name)}\.lora_{matrix}\.weight$"
    )
    matches = [k for k in keys if pattern.search(k)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        return None
    raise ValueError(
        f"ambiguous lora key match for layer={layer} module={hf_module_name} "
        f"matrix={matrix}: {matches}"
    )


def write_f16_block(f, arr: np.ndarray):
    f.write(arr.astype(np.float32).astype(np.float16).tobytes())


def main():
    parser = ArgumentParser(
        description="Convert a HuggingFace PEFT LoRA adapter to project-zero's .lora.bin format"
    )
    parser.add_argument("--adapter", type=str, required=True,
                         help="HF PEFT adapter directory (adapter_config.json + adapter_model.safetensors)")
    parser.add_argument("--output", type=str, required=True, help="Output .lora.bin path")
    parser.add_argument("--dim", type=int, required=True,
                         help="Base model's attention dim (Config.dim — see the base GGUF's startup banner)")
    parser.add_argument("--hidden-dim", type=int, required=True,
                         help="Base model's FFN hidden_dim (Config.hidden_dim)")
    parser.add_argument("--n-layers", type=int, required=True,
                         help="Base model's layer count (Config.n_layers)")
    parser.add_argument("--kv-dim", type=int, default=None,
                         help="Base model's GQA kv_dim (n_kv_heads * head_dim — printed as "
                              "'kv_dim' in the base GGUF's startup banner). Required when the "
                              "adapter targets k_proj/v_proj on a GQA model (n_kv_heads < "
                              "n_heads) — K/V project to kv_dim, not dim, and a wrong value "
                              "here corrupts every K/V-targeting LoRA correction silently. "
                              "Defaults to --dim (correct only for a non-GQA/MHA base model).")
    args = parser.parse_args()
    kv_dim = args.kv_dim if args.kv_dim is not None else args.dim
    if args.kv_dim is None and ({"k_proj", "v_proj"} & set(json.load(
            open(os.path.join(args.adapter, "adapter_config.json"))).get("target_modules", []))):
        print(f"warning: adapter targets k_proj/v_proj but --kv-dim was not given — "
              f"defaulting to --kv-dim={args.dim} (only correct if the base model uses "
              f"plain multi-head attention, i.e. n_kv_heads == n_heads). Pass --kv-dim "
              f"explicitly for a GQA base model (check the GGUF's startup banner).")

    config_path = os.path.join(args.adapter, "adapter_config.json")
    with open(config_path) as fh:
        cfg = json.load(fh)

    rank = int(cfg["r"])
    alpha = float(cfg.get("lora_alpha", rank))
    hf_target_modules = set(cfg.get("target_modules", []))

    unsupported = hf_target_modules - {name for name, _ in TARGETS}
    if unsupported:
        print(f"warning: adapter targets modules this converter/engine doesn't "
              f"support (Phase 19 scope: generic dense/GQA attention+FFN only): "
              f"{sorted(unsupported)} — skipping them")

    target_mask = 0
    for hf_name, bit in TARGETS:
        if hf_name in hf_target_modules:
            target_mask |= (1 << bit)

    if target_mask == 0:
        raise ValueError(
            f"none of this adapter's target_modules ({sorted(hf_target_modules)}) "
            "are supported by this engine's Phase 19 scope "
            "(q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj)"
        )

    weights_path = find_adapter_weights_file(args.adapter)

    print(f"adapter: {args.adapter}")
    print(f"  rank={rank} alpha={alpha} scale={alpha/rank:.4f}")
    print(f"  target_modules={sorted(hf_target_modules)} -> mask=0x{target_mask:02x}")
    print(f"  base model shape: dim={args.dim} kv_dim={kv_dim} hidden_dim={args.hidden_dim} n_layers={args.n_layers}")

    with safe_open(weights_path, framework="numpy", device="cpu") as st:
        keys = list(st.keys())

        with open(args.output, "wb") as out:
            out.write(struct.pack("<I", LORA_BIN_MAGIC))
            out.write(struct.pack("<I", LORA_BIN_VERSION))
            out.write(struct.pack("<i", args.n_layers))
            out.write(struct.pack("<i", rank))
            out.write(struct.pack("<f", alpha))
            out.write(struct.pack("<i", args.dim))
            out.write(struct.pack("<i", args.hidden_dim))
            out.write(struct.pack("<i", target_mask))
            out.write(struct.pack("<i", kv_dim))
            out.write(b"\x00" * 28)  # reserved, pad header to 64 bytes
            assert out.tell() == 64

            for hf_name, bit in TARGETS:
                if not (target_mask & (1 << bit)):
                    continue
                print(f"  writing target '{hf_name}'...")
                for layer in range(args.n_layers):
                    key_a = find_lora_key(keys, layer, hf_name, "A")
                    key_b = find_lora_key(keys, layer, hf_name, "B")
                    if key_a is None or key_b is None:
                        raise ValueError(
                            f"target_modules includes '{hf_name}' but no lora_A/lora_B "
                            f"tensor found for layer {layer} (looked for '.layers.{layer}.'"
                            f" + '{hf_name}.lora_A/B.weight')"
                        )
                    a = st.get_tensor(key_a)  # [rank, in_features], PEFT-native — no transpose
                    b = st.get_tensor(key_b)  # [out_features, rank], PEFT-native — no transpose
                    write_f16_block(out, a)
                    write_f16_block(out, b)

    size = os.path.getsize(args.output)
    print(f"wrote {args.output} ({size} bytes)")


if __name__ == "__main__":
    main()
