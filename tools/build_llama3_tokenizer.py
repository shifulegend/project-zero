import json
import struct
import sys
import os

def build_tokenizer():
    input_path = "/tmp/tokenizer.json"
    output_path = "models/bitnet-b1.58-2B-4T_tokenizer_proper.bin"
    
    with open(input_path, "r") as f:
        data = json.load(f)
    
    # Llama-3 (Tiktoken) tokenizer.json structure:
    # model.vocab contains { token_string: id }
    # added_tokens contains list of special tokens
    
    model = data.get("model", {})
    vocab = model.get("vocab", {})
    added_tokens = data.get("added_tokens", [])
    
    # Build complete mapping
    full_vocab = {}
    for token, idx in vocab.items():
        full_vocab[idx] = token
    
    for entry in added_tokens:
        idx = entry["id"]
        content = entry["content"]
        full_vocab[idx] = content
    
    vocab_size = max(full_vocab.keys()) + 1
    print(f"Detected vocab size: {vocab_size}")
    
    # We'll use id-based scores (engine picks max score, so -id works if merges are ordered)
    # Actually, Llama-3 merges are roughly in-order.
    
    max_token_len = 0
    
    with open(output_path, "wb") as f:
        # We need to know max_token_len beforehand or just use a safe buffer
        # Let's pre-calculate
        for i in range(vocab_size):
            t = full_vocab.get(i, f"<token_{i}>")
            # Convert to bytes (UTF-8)
            b = t.encode("utf-8")
            max_token_len = max(max_token_len, len(b))
            
        f.write(struct.pack('<i', vocab_size))
        f.write(struct.pack('<i', max_token_len))
        
        for i in range(vocab_size):
            t = full_vocab.get(i, f"<token_{i}>")
            b = t.encode("utf-8")
            score = -float(i)
            f.write(struct.pack('<f', score))
            f.write(struct.pack('<i', len(b)))
            f.write(b)

if __name__ == "__main__":
    build_tokenizer()
