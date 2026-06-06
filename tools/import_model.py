#!/usr/bin/env python3
"""
import_model.py — Phase 34.1: Universal Model Importer

Auto-detects the model type from a HuggingFace repo or local directory,
then routes to the appropriate converter to produce .bin files for the
Project Zero C engine.

Detection pipeline:
  1. Reads config.json → model_type
  2. If ternary BitNet weights (uint8, {0,1,2} values) → convert_hf_bitnet.py
  3. If multimodal (has vision encoder) → extract_multimodal.py
  4. If standard float16/BF16 → convert_hf.py (text-only quantization)

Usage:
  python import_model.py --repo microsoft/BitNet-b1.58-2B-4T --out models/
  python import_model.py --repo moondream-hf/moondream2       --out models/
  python import_model.py --repo Qwen/Qwen2.5-7B-Instruct      --out models/
  python import_model.py --local /path/to/model --out models/
"""

import argparse
import json
import os
import subprocess
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))


def run(script, extra_args):
    cmd = [sys.executable, os.path.join(TOOLS_DIR, script)] + extra_args
    print(f'\n  → Running: {" ".join(cmd)}\n')
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f'ERROR: {script} failed (exit {result.returncode})')
        sys.exit(result.returncode)


def detect_model_class(config, model_dir):
    """
    Returns one of: 'ternary_bitnet', 'multimodal', 'standard_float'
    """
    model_type = config.get('model_type', '').lower()
    arch       = config.get('architectures', [''])[0].lower()

    # Check for known multimodal indicators
    multimodal_types = {'llava', 'moondream', 'idefics', 'smolvlm',
                        'qwen2_vl', 'paligemma', 'internvl', 'cogvlm'}
    if any(t in model_type for t in multimodal_types) or \
       any(t in arch       for t in multimodal_types):
        return 'multimodal'

    # Check for BitNet-style quantized_ternary model
    quantization = config.get('quantization_config', {})
    quant_type   = quantization.get('quant_type', '').lower()
    if 'bitnet' in quant_type or 'ternary' in quant_type or \
       'bitnet' in model_type or '1.58' in config.get('model_name', ''):
        return 'ternary_bitnet'

    # Inspect actual weights if safetensors available
    try:
        from safetensors import safe_open
        import numpy as np
        import glob
        sf = sorted(glob.glob(os.path.join(model_dir, '*.safetensors')))
        if sf:
            with safe_open(sf[0], framework='numpy') as f:
                keys = list(f.keys())
                # Sample first weight tensor
                wkeys = [k for k in keys if 'weight' in k and 'scale' not in k
                         and 'norm' not in k and 'embed' not in k]
                if wkeys:
                    sample = f.get_tensor(wkeys[0])
                    if sample.dtype == np.uint8:
                        unique = set(sample.flatten().tolist())
                        # Ternary packed: values are combinations of {0,1,2} per 2-bit nibble
                        if unique.issubset({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}):
                            return 'ternary_bitnet'
    except Exception:
        pass

    return 'standard_float'


def load_config(path):
    config_path = os.path.join(path, 'config.json')
    if not os.path.exists(config_path):
        return {}
    with open(config_path) as fh:
        return json.load(fh)


def hf_download(repo_id, cache_dir):
    """Download model to HF cache, return local directory."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print('ERROR: pip install huggingface_hub')
        sys.exit(1)
    print(f'Downloading {repo_id} ...')
    return snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        ignore_patterns=['*.bin', '*.pt', 'optimizer*'],
    )


def main():
    parser = argparse.ArgumentParser(description='Universal model importer for Project Zero')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--repo',  help='HuggingFace repo ID')
    group.add_argument('--local', help='Local model directory')
    parser.add_argument('--out',   default='models/', help='Output directory')
    parser.add_argument('--cache', default=None,      help='HF cache dir')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print('Phase 34.1 — Universal Model Importer')
    print('======================================')

    if args.repo:
        model_dir = hf_download(args.repo, args.cache)
        source_arg = ['--repo', args.repo]
    else:
        model_dir  = args.local
        source_arg = ['--local', args.local]

    config = load_config(model_dir)
    cls    = detect_model_class(config, model_dir)

    print(f'Model type detected: {cls}')
    print(f'Model dir: {model_dir}')

    out_args = ['--out', args.out]
    if args.cache:
        out_args += ['--cache', args.cache]

    if cls == 'ternary_bitnet':
        print('Route → convert_hf_bitnet.py (ternary packing)')
        run('convert_hf_bitnet.py', ['--model', model_dir, '--output',
                                      os.path.join(args.out, 'model.bin')])

    elif cls == 'multimodal':
        print('Route → extract_multimodal.py (vision encoder + projector)')
        run('extract_multimodal.py', source_arg + out_args)
        print('\nNote: multimodal LLM text backbone conversion not yet supported.')
        print('      vision.bin and projector.bin extracted for use with --vision/--proj flags.')

    else:  # standard_float
        print('Route → convert_hf.py (float16 → standard .bin)')
        run('convert_hf.py', ['--model', model_dir,
                               '--output', os.path.join(args.out, 'model.bin')])


if __name__ == '__main__':
    main()
