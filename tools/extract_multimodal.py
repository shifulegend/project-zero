#!/usr/bin/env python3
"""
extract_multimodal.py — Phase 34.3: Multimodal Model Weight Extractor

Extracts vision encoder and MLP projector weights from a HuggingFace
multimodal model and writes them in the Project Zero binary format so
the C engine can load them with vision_weights_load.c.

Supported models (auto-detected from config.json):
  - Moondream2      (moondream-hf/moondream2)         SigLIP-400M + Phi-1.5
  - SmolVLM-256M   (HuggingFaceTB/SmolVLM-256M-Instruct)   SigLIP + SmolLM
  - SmolVLM-500M   (HuggingFaceTB/SmolVLM-500M-Instruct)   SigLIP + SmolLM
  - LLaVA-1.5      (llava-hf/llava-1.5-7b-hf)         CLIP + Vicuna
  - Qwen2-VL-*     (Qwen/Qwen2-VL-*)                  SigLIP + Qwen2

Usage:
  python extract_multimodal.py \\
      --repo  moondream-hf/moondream2 \\
      --out   models/

  python extract_multimodal.py \\
      --local /path/to/model/dir \\
      --out   models/

Outputs:
  models/vision.bin      — SigLIP/ViT encoder weights (float32)
  models/projector.bin   — MLP projector weights (float32)

Binary format is documented in include/multimodal/vision_weights_load.h.
"""

import argparse
import json
import math
import os
import struct
import sys

import numpy as np

VISION_MAGIC    = 0x57535956  # "VISW" LE
PROJECTOR_MAGIC = 0x4A4F5250  # "PROJ" LE
VERSION         = 1
HEADER_SIZE     = 64  # padded to 64 bytes


# ── Binary write helpers ────────────────────────────────────────────────────

def write_header_vision(f, n_layers, embed_dim, hidden_dim, n_heads, patch_dim, num_patches):
    hdr = struct.pack('<IIiiiiii',
                      VISION_MAGIC, VERSION,
                      n_layers, embed_dim, hidden_dim,
                      n_heads, patch_dim, num_patches)
    f.write(hdr)
    f.write(b'\x00' * (HEADER_SIZE - len(hdr)))


def write_header_proj(f, vision_dim, llm_dim, hidden_dim, has_bias, scale_factor=1):
    # [0:24] magic, version, vision_dim, llm_dim, hidden_dim, has_bias
    # [24:28] scale_factor (new field; 1 = 2-layer MLP, >1 = pixel-shuffle + single linear)
    # [28:64] reserved
    hdr = struct.pack('<IIiiiiii',
                      PROJECTOR_MAGIC, VERSION,
                      vision_dim, llm_dim, hidden_dim, has_bias, scale_factor, 0)
    f.write(hdr)
    f.write(b'\x00' * (HEADER_SIZE - len(hdr)))


def write_f32(f, tensor):
    """Write a tensor as float32 (convert from bf16/f16 if needed)."""
    arr = np.array(tensor, dtype=np.float32)
    f.write(arr.tobytes())


# ── Model-specific extractors ───────────────────────────────────────────────

class ModelAdapter:
    """Base class — subclasses implement get_vision_tensors / get_proj_tensors."""

    def __init__(self, tensors, config):
        self.tensors = tensors  # dict: name -> numpy array
        self.config  = config

    def get(self, name):
        t = self.tensors.get(name)
        if t is None:
            raise KeyError(f"tensor not found: {name!r}")
        return t

    def vision_config(self):
        """Return (n_layers, embed_dim, hidden_dim, n_heads, patch_size)."""
        raise NotImplementedError

    def write_vision_weights(self, f):
        raise NotImplementedError

    def write_proj_weights(self, f):
        raise NotImplementedError


class Moondream2Adapter(ModelAdapter):
    """
    Moondream2 uses a SigLIP-style encoder. Tensor prefix: 'vision_encoder.'
    Projector tensors: 'vision_projection.' (linear layers w/o bias in some versions)
    """

    PREFIX_ENC  = 'vision_encoder.encoder.layers.'
    PREFIX_PROJ = 'vision_projection.'

    def vision_config(self):
        vcfg = self.config.get('vision_config', self.config)
        n_layers   = vcfg.get('num_hidden_layers', 27)
        embed_dim  = vcfg.get('hidden_size', 1152)
        hidden_dim = vcfg.get('intermediate_size', 4304)
        n_heads    = vcfg.get('num_attention_heads', 16)
        patch_size = vcfg.get('patch_size', 14)
        return n_layers, embed_dim, hidden_dim, n_heads, patch_size

    def _layer_key(self, l, sub):
        return f'{self.PREFIX_ENC}{l}.{sub}'

    def write_vision_weights(self, f):
        n_layers, embed_dim, hidden_dim, n_heads, patch_size = self.vision_config()
        patch_dim   = patch_size * patch_size * 3
        num_patches = (384 // patch_size) ** 2

        write_header_vision(f, n_layers, embed_dim, hidden_dim, n_heads, patch_dim, num_patches)

        # patch embedding projection
        write_f32(f, self.get('vision_encoder.patch_emb.weight')
                      .reshape(embed_dim, patch_dim))
        # SigLIP has no patch_proj_b in some versions — use zeros
        try:
            write_f32(f, self.get('vision_encoder.patch_emb.bias'))
        except KeyError:
            write_f32(f, np.zeros(embed_dim, dtype=np.float32))

        # positional embedding
        try:
            write_f32(f, self.get('vision_encoder.pos_emb'))
        except KeyError:
            write_f32(f, np.zeros((num_patches, embed_dim), dtype=np.float32))

        # final layernorm weight
        write_f32(f, self.get('vision_encoder.encoder.final_layer_norm.weight'))

        # per-layer weights
        for l in range(n_layers):
            write_f32(f, self.get(self._layer_key(l, 'layer_norm1.weight')))
            write_f32(f, self.get(self._layer_key(l, 'layer_norm2.weight')))
            write_f32(f, self.get(self._layer_key(l, 'self_attn.q_proj.weight')))
            write_f32(f, self.get(self._layer_key(l, 'self_attn.k_proj.weight')))
            write_f32(f, self.get(self._layer_key(l, 'self_attn.v_proj.weight')))
            write_f32(f, self.get(self._layer_key(l, 'self_attn.out_proj.weight')))
            write_f32(f, self.get(self._layer_key(l, 'mlp.fc1.weight')))
            # w2 (down): [embed_dim × hidden_dim]
            write_f32(f, self.get(self._layer_key(l, 'mlp.fc2.weight')))
            # w3 (gate/up): zeros (SigLIP uses single-gate MLP, not gated)
            write_f32(f, np.zeros((hidden_dim, embed_dim), dtype=np.float32))

    def write_proj_weights(self, f):
        n_layers, embed_dim, _, _, _ = self.vision_config()
        text_cfg   = self.config.get('text_config', self.config)
        llm_dim    = text_cfg.get('hidden_size', 2048)

        # Moondream2 projector: fc1 + fc2 (2-layer MLP)
        w1_key = f'{self.PREFIX_PROJ}0.weight'
        w2_key = f'{self.PREFIX_PROJ}2.weight'
        has_b1 = f'{self.PREFIX_PROJ}0.bias' in self.tensors
        has_b2 = f'{self.PREFIX_PROJ}2.bias' in self.tensors
        has_bias = int(has_b1 and has_b2)

        w1 = self.get(w1_key)
        w2 = self.get(w2_key)
        hidden_dim = w1.shape[0]

        write_header_proj(f, embed_dim, llm_dim, hidden_dim, has_bias)
        write_f32(f, w1)
        if has_bias: write_f32(f, self.get(f'{self.PREFIX_PROJ}0.bias'))
        write_f32(f, w2)
        if has_bias: write_f32(f, self.get(f'{self.PREFIX_PROJ}2.bias'))


class SmolVLMAdapter(ModelAdapter):
    """
    SmolVLM uses SigLIP encoder. Tensor prefix: 'model.vision_model.'
    Projector prefix: 'model.connector.'
    """

    PREFIX_ENC  = 'model.vision_model.encoder.layers.'
    PREFIX_PROJ = 'model.connector.'

    def vision_config(self):
        vcfg = self.config.get('vision_config', {})
        n_layers   = vcfg.get('num_hidden_layers', 27)
        embed_dim  = vcfg.get('hidden_size', 1152)
        hidden_dim = vcfg.get('intermediate_size', 4304)
        n_heads    = vcfg.get('num_attention_heads', 16)
        patch_size = vcfg.get('patch_size', 14)
        return n_layers, embed_dim, hidden_dim, n_heads, patch_size

    def write_vision_weights(self, f):
        n_layers, embed_dim, hidden_dim, n_heads, patch_size = self.vision_config()
        patch_dim   = patch_size * patch_size * 3
        num_patches = (384 // patch_size) ** 2

        write_header_vision(f, n_layers, embed_dim, hidden_dim, n_heads, patch_dim, num_patches)

        pfx = 'model.vision_model.'
        write_f32(f, self.get(f'{pfx}embeddings.patch_embedding.weight')
                      .reshape(embed_dim, patch_dim))
        try:
            write_f32(f, self.get(f'{pfx}embeddings.patch_embedding.bias'))
        except KeyError:
            write_f32(f, np.zeros(embed_dim, dtype=np.float32))

        try:
            write_f32(f, self.get(f'{pfx}embeddings.position_embedding.weight'))
        except KeyError:
            write_f32(f, np.zeros((num_patches, embed_dim), dtype=np.float32))

        write_f32(f, self.get(f'{pfx}post_layernorm.weight'))

        p = self.PREFIX_ENC
        for l in range(n_layers):
            write_f32(f, self.get(f'{p}{l}.layer_norm1.weight'))
            write_f32(f, self.get(f'{p}{l}.layer_norm2.weight'))
            write_f32(f, self.get(f'{p}{l}.self_attn.q_proj.weight'))
            write_f32(f, self.get(f'{p}{l}.self_attn.k_proj.weight'))
            write_f32(f, self.get(f'{p}{l}.self_attn.v_proj.weight'))
            write_f32(f, self.get(f'{p}{l}.self_attn.out_proj.weight'))
            write_f32(f, self.get(f'{p}{l}.mlp.fc1.weight'))
            write_f32(f, self.get(f'{p}{l}.mlp.fc2.weight'))
            write_f32(f, np.zeros((hidden_dim, embed_dim), dtype=np.float32))

    def write_proj_weights(self, f):
        n_layers, embed_dim, _, _, _ = self.vision_config()
        tcfg       = self.config.get('text_config', {})
        llm_dim    = tcfg.get('hidden_size', 576)
        scale_factor = self.config.get('scale_factor', 1)

        single_key = f'{self.PREFIX_PROJ}modality_projection.proj.weight'
        mlp0_key   = f'{self.PREFIX_PROJ}modality_projection.proj.0.weight'

        if single_key in self.tensors:
            # Single-linear + pixel-shuffle connector (SmolVLM-256M / newer Idefics3)
            # The weight maps  (embed_dim * scale_factor^2) → llm_dim.
            w = self.get(single_key)   # [llm_dim, embed_dim * scale^2]
            has_b = int(f'{self.PREFIX_PROJ}modality_projection.proj.bias' in self.tensors)
            # hidden_dim=0 is the sentinel for "single-linear" mode in the C engine
            write_header_proj(f, embed_dim, llm_dim, 0, has_b, scale_factor)
            write_f32(f, w)
            if has_b:
                write_f32(f, self.get(f'{self.PREFIX_PROJ}modality_projection.proj.bias'))
        elif mlp0_key in self.tensors:
            # Two-layer MLP connector (SmolVLM-500M and larger variants)
            w1 = self.get(mlp0_key)
            w2 = self.get(f'{self.PREFIX_PROJ}modality_projection.proj.2.weight')
            has_b = int(mlp0_key.replace('0.weight', '0.bias') in self.tensors)
            hidden_dim = w1.shape[0]
            write_header_proj(f, embed_dim, llm_dim, hidden_dim, has_b, scale_factor=1)
            write_f32(f, w1)
            if has_b:
                write_f32(f, self.get(f'{self.PREFIX_PROJ}modality_projection.proj.0.bias'))
            write_f32(f, w2)
            if has_b:
                write_f32(f, self.get(f'{self.PREFIX_PROJ}modality_projection.proj.2.bias'))
        else:
            raise KeyError(
                f"Cannot find connector weights: tried {single_key!r} and {mlp0_key!r}\n"
                f"Available connector keys: {[k for k in self.tensors if 'connector' in k]}"
            )


class LLaVAAdapter(ModelAdapter):
    """LLaVA-1.5: CLIP ViT-L/14 encoder + MLP projector."""

    PREFIX_ENC  = 'model.vision_tower.vision_tower.vision_model.encoder.layers.'
    PREFIX_PROJ = 'model.mm_projector.'

    def vision_config(self):
        vcfg = self.config.get('mm_vision_tower', 'openai/clip-vit-large-patch14-336')
        # ViT-L/14: 24 layers, 1024 dim, 4096 hidden, 16 heads, patch=14
        return 24, 1024, 4096, 16, 14

    def write_vision_weights(self, f):
        n_layers, embed_dim, hidden_dim, n_heads, patch_size = self.vision_config()
        patch_dim   = patch_size * patch_size * 3
        num_patches = (336 // patch_size) ** 2  # LLaVA uses 336px

        write_header_vision(f, n_layers, embed_dim, hidden_dim, n_heads, patch_dim, num_patches)

        pfx = 'model.vision_tower.vision_tower.vision_model.'
        write_f32(f, self.get(f'{pfx}embeddings.patch_embedding.weight')
                      .reshape(embed_dim, patch_dim))
        write_f32(f, np.zeros(embed_dim, dtype=np.float32))  # no bias
        write_f32(f, self.get(f'{pfx}embeddings.position_embedding.weight')[1:, :])
        write_f32(f, self.get(f'{pfx}post_layernorm.weight'))

        p = self.PREFIX_ENC
        for l in range(n_layers):
            write_f32(f, self.get(f'{p}{l}.layer_norm1.weight'))
            write_f32(f, self.get(f'{p}{l}.layer_norm2.weight'))
            write_f32(f, self.get(f'{p}{l}.self_attn.q_proj.weight'))
            write_f32(f, self.get(f'{p}{l}.self_attn.k_proj.weight'))
            write_f32(f, self.get(f'{p}{l}.self_attn.v_proj.weight'))
            write_f32(f, self.get(f'{p}{l}.self_attn.out_proj.weight'))
            write_f32(f, self.get(f'{p}{l}.mlp.fc1.weight'))
            write_f32(f, self.get(f'{p}{l}.mlp.fc2.weight'))
            write_f32(f, np.zeros((hidden_dim, embed_dim), dtype=np.float32))

    def write_proj_weights(self, f):
        _, embed_dim, _, _, _ = self.vision_config()
        tcfg    = self.config.get('text_config', {})
        llm_dim = tcfg.get('hidden_size', 4096)

        w1 = self.get(f'{self.PREFIX_PROJ}0.weight')
        w2 = self.get(f'{self.PREFIX_PROJ}2.weight')
        has_b = int(f'{self.PREFIX_PROJ}0.bias' in self.tensors)
        hidden_dim = w1.shape[0]

        write_header_proj(f, embed_dim, llm_dim, hidden_dim, has_b)
        write_f32(f, w1)
        if has_b: write_f32(f, self.get(f'{self.PREFIX_PROJ}0.bias'))
        write_f32(f, w2)
        if has_b: write_f32(f, self.get(f'{self.PREFIX_PROJ}2.bias'))


# ── Adapter registry ─────────────────────────────────────────────────────────

def detect_adapter(config, tensors):
    model_type = config.get('model_type', '').lower()
    arch       = config.get('architectures', [''])[0].lower()

    if 'moondream' in model_type or 'moondream' in arch:
        print(f'  Detected: Moondream2')
        return Moondream2Adapter(tensors, config)
    if 'smolvlm' in model_type or 'idefics3' in model_type or 'smolvlm' in arch:
        print(f'  Detected: SmolVLM')
        return SmolVLMAdapter(tensors, config)
    if 'llava' in model_type or 'llava' in arch:
        print(f'  Detected: LLaVA-1.5')
        return LLaVAAdapter(tensors, config)

    # Fallback heuristic: try SmolVLM layout
    if any('vision_model' in k for k in tensors):
        print(f'  Detected: Generic SigLIP-style (SmolVLM layout assumed)')
        return SmolVLMAdapter(tensors, config)

    raise ValueError(
        f"Unsupported model type: {model_type!r} / arch: {arch!r}\n"
        "Supported: moondream2, smolvlm, llava-1.5\n"
        "Open a PR to add support for other models."
    )


# ── Tensor loading ────────────────────────────────────────────────────────────

def load_tensors_from_dir(model_dir):
    """Load all tensors from safetensors files in model_dir."""
    try:
        from safetensors import safe_open
    except ImportError:
        print("ERROR: safetensors not installed. Run: pip install safetensors")
        sys.exit(1)

    import glob
    sf_files = sorted(glob.glob(os.path.join(model_dir, '*.safetensors')))
    if not sf_files:
        raise FileNotFoundError(f"No .safetensors files in {model_dir}")

    tensors = {}
    for fpath in sf_files:
        with safe_open(fpath, framework='pt') as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key).float().numpy()
        print(f'  Loaded {os.path.basename(fpath)} ({len(tensors)} tensors total)')

    return tensors


def load_tensors_from_hf(repo_id, cache_dir=None):
    """Download and load tensors from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    print(f'  Downloading {repo_id} from HuggingFace...')
    local_dir = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        ignore_patterns=['*.bin', '*.pt', 'optimizer*', 'training*'],
    )
    print(f'  Downloaded to {local_dir}')

    config_path = os.path.join(local_dir, 'config.json')
    with open(config_path) as fh:
        config = json.load(fh)

    tensors = load_tensors_from_dir(local_dir)
    return tensors, config, local_dir


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Extract vision encoder + projector from a multimodal HF model'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--repo',  help='HuggingFace repo ID (e.g. moondream-hf/moondream2)')
    group.add_argument('--local', help='Path to local model directory')
    parser.add_argument('--out',  default='models/', help='Output directory (default: models/)')
    parser.add_argument('--cache', default=None,     help='HF cache directory')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print('Phase 34.3 — Multimodal Weight Extractor')
    print('=========================================')

    if args.repo:
        tensors, config, _ = load_tensors_from_hf(args.repo, cache_dir=args.cache)
    else:
        config_path = os.path.join(args.local, 'config.json')
        if not os.path.exists(config_path):
            print(f'ERROR: no config.json in {args.local}')
            sys.exit(1)
        with open(config_path) as fh:
            config = json.load(fh)
        tensors = load_tensors_from_dir(args.local)

    print(f'\nTotal tensors loaded: {len(tensors)}')
    total_params = sum(t.size for t in tensors.values())
    print(f'Total parameters: {total_params / 1e6:.1f}M')

    adapter = detect_adapter(config, tensors)

    # -- Write vision.bin --
    vision_out = os.path.join(args.out, 'vision.bin')
    print(f'\nWriting {vision_out} ...')
    with open(vision_out, 'wb') as f:
        adapter.write_vision_weights(f)
    vision_size = os.path.getsize(vision_out)
    print(f'  Done: {vision_size / 1e6:.1f} MB')

    # -- Write projector.bin --
    proj_out = os.path.join(args.out, 'projector.bin')
    print(f'Writing {proj_out} ...')
    with open(proj_out, 'wb') as f:
        adapter.write_proj_weights(f)
    proj_size = os.path.getsize(proj_out)
    print(f'  Done: {proj_size / 1e6:.1f} MB')

    n_layers, embed_dim, hidden_dim, n_heads, patch_size = adapter.vision_config()
    print(f'\nSummary:')
    print(f'  Vision encoder: {n_layers}L × {embed_dim}d × {n_heads}h, patch={patch_size}px')
    print(f'  Run with:')
    print(f'    ./adaptive_ai_engine \\')
    print(f'        --model  models/bitnet-b1.58-2B-4T.bin \\')
    print(f'        --tokenizer models/bitnet-b1.58-2B-4T_tokenizer_proper.bin \\')
    print(f'        --vision {vision_out} \\')
    print(f'        --proj   {proj_out} \\')
    print(f'        --image  strawberry.jpg \\')
    print(f'        --prompt "What do you see in this image?"')


if __name__ == '__main__':
    main()
