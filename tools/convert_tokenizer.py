#!/usr/bin/env python3
"""
convert_tokenizer.py — Tokenizer Converter

Phase 10.3: Converts HuggingFace tokenizer.json or SentencePiece .model
files into the C engine's binary vocab format.

Binary format (matching tokenizer_load.c):
  [vocab_size: int32]
  [max_token_len: int32]
  For each token:
    [score: float32]
    [len: int32]
    [bytes: char[len]]

Usage:
  python convert_tokenizer.py --input tokenizer.json --output tokenizer.bin
"""

import argparse
import json
import struct
import os
import sys


def convert_hf_tokenizer(input_path, output_path):
    """Convert HuggingFace tokenizer.json to binary format."""

    with open(input_path, 'r', encoding='utf-8') as f:
        tok_data = json.load(f)

    # Extract vocabulary
    vocab = {}
    if 'model' in tok_data and 'vocab' in tok_data['model']:
        vocab = tok_data['model']['vocab']
    elif 'added_tokens' in tok_data:
        # Some tokenizers only have added_tokens
        for entry in tok_data['added_tokens']:
            vocab[entry['content']] = entry['id']

    if not vocab:
        print("Error: Could not find vocabulary in tokenizer.json",
              file=sys.stderr)
        sys.exit(1)

    # Also include added_tokens (e.g. BOS/EOS special tokens with IDs >= base vocab size)
    if 'added_tokens' in tok_data:
        existing_ids = set(vocab.values())
        for entry in tok_data['added_tokens']:
            token_id = entry['id']
            content = entry['content']
            if token_id not in existing_ids:
                vocab[content] = token_id
                existing_ids.add(token_id)

    # Extract merge scores if available (BPE models)
    merges = []
    if 'model' in tok_data and 'merges' in tok_data['model']:
        merges = tok_data['model']['merges']

    vocab_size = max(vocab.values()) + 1
    max_token_len = max(len(token.encode('utf-8')) for token in vocab)

    # Build token array indexed by ID
    tokens = [''] * vocab_size
    scores = [0.0] * vocab_size

    for token_str, token_id in vocab.items():
        if 0 <= token_id < vocab_size:
            tokens[token_id] = token_str
            # Score: negative of merge rank (lower rank = higher priority)
            scores[token_id] = 0.0

    # Assign scores from merge order (if BPE)
    merge_vocab = {}
    for i, merge in enumerate(merges):
        merged_token = (merge if isinstance(merge, str) else ''.join(merge)).replace(' ', '')
        if merged_token in vocab:
            # Higher priority merges get higher (less negative) scores
            scores[vocab[merged_token]] = -(i + 1)

    # Write binary file
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<i', vocab_size))
        f.write(struct.pack('<i', max_token_len))

        for i in range(vocab_size):
            token_bytes = tokens[i].encode('utf-8')
            f.write(struct.pack('<f', scores[i]))
            f.write(struct.pack('<i', len(token_bytes)))
            f.write(token_bytes)

    file_size = os.path.getsize(output_path)
    print(f"Converted tokenizer: {output_path}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Max token len: {max_token_len}")
    print(f"  File size: {file_size:,} bytes")


def convert_sentencepiece(input_path, output_path):
    """Convert SentencePiece .model to binary format."""
    try:
        import sentencepiece as spm
    except ImportError:
        print("Error: sentencepiece package required for .model files",
              file=sys.stderr)
        print("Install with: pip install sentencepiece", file=sys.stderr)
        sys.exit(1)

    sp = spm.SentencePieceProcessor()
    sp.load(input_path)

    vocab_size = sp.get_piece_size()
    max_token_len = 0

    tokens = []
    scores = []
    for i in range(vocab_size):
        piece = sp.id_to_piece(i)
        score = sp.get_score(i)
        piece_bytes = piece.encode('utf-8')
        max_token_len = max(max_token_len, len(piece_bytes))
        tokens.append(piece_bytes)
        scores.append(score)

    with open(output_path, 'wb') as f:
        f.write(struct.pack('<i', vocab_size))
        f.write(struct.pack('<i', max_token_len))

        for i in range(vocab_size):
            f.write(struct.pack('<f', scores[i]))
            f.write(struct.pack('<i', len(tokens[i])))
            f.write(tokens[i])

    file_size = os.path.getsize(output_path)
    print(f"Converted tokenizer: {output_path}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Max token len: {max_token_len}")
    print(f"  File size: {file_size:,} bytes")


def main():
    parser = argparse.ArgumentParser(
        description='Convert tokenizer to binary format')
    parser.add_argument('--input', required=True,
                        help='Input tokenizer.json or .model file')
    parser.add_argument('--output', required=True,
                        help='Output binary file path')
    args = parser.parse_args()

    if args.input.endswith('.json'):
        convert_hf_tokenizer(args.input, args.output)
    elif args.input.endswith('.model'):
        convert_sentencepiece(args.input, args.output)
    else:
        print("Error: Input must be .json or .model file", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
