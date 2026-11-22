#!/usr/bin/env python3
"""Writes a tiny random native ternary model and a matching legacy tokenizer.

The native format has no converter in this repository's tests, so this builds one from the layout
`weights_map` (src/runtime/weights.c) reads: a 64-byte header (magic, version, the 40-byte config),
then the BF16 embedding table, then per layer the norms, the 2-bit packed ternary projections each
followed by its float scale, and the sub-norms, then the final norm; everything 64-byte aligned.
The weights are random, so the model's output is nonsense - it exists so that `load_native`, the
legacy tokenizer loader and a short generation have something real to run on, deterministically.

Usage: tests/make_tiny_native.py <out-dir> [--moe]    (writes model.bin and tokenizer.bin)
"""
import os
import random
import struct
import sys

DIM, HIDDEN, LAYERS, HEADS, KV_HEADS, VOCAB, SEQ = 64, 128, 2, 4, 2, 128, 128
TN_MAGIC, TN_VERSION = 0x594E5254, 1


class Writer:
    """Bytes of the weight section, tracking the absolute offset (the section starts at 64)."""

    def __init__(self):
        self.buf = bytearray()

    def raw(self, data):
        self.buf += data

    def align64(self):
        self.buf += b"\0" * ((64 - (len(self.buf) + 64) % 64) % 64)


def packed(rng, count):
    """`count` ternary weights as 2-bit codes 0..2, four to a byte (LSB first)."""
    nbytes = (count + 3) >> 2
    return bytes(
        sum(rng.randrange(3) << (2 * j) for j in range(4)) for _ in range(nbytes)
    )


def floats(rng, n, lo, hi):
    return struct.pack("<%df" % n, *[rng.uniform(lo, hi) for _ in range(n)])


def bf16_table(rng, n):
    out = bytearray()
    for _ in range(n):
        f = rng.uniform(-0.5, 0.5)
        out += struct.pack("<I", struct.unpack("<I", struct.pack("<f", f))[0])[2:]
    return bytes(out)


def projection(w, rng, count):
    w.raw(packed(rng, count))
    w.raw(struct.pack("<f", rng.uniform(0.02, 0.1)))
    w.align64()


def build_model(rng):
    w = Writer()
    kv_dim = DIM // HEADS * KV_HEADS
    w.raw(bf16_table(rng, VOCAB * DIM))
    w.align64()
    for _ in range(LAYERS):
        w.raw(floats(rng, DIM, 0.8, 1.2))
        w.align64()
        w.raw(floats(rng, DIM, 0.8, 1.2))
        w.align64()
        projection(w, rng, DIM * DIM)  # wq
        projection(w, rng, DIM * kv_dim)  # wk
        projection(w, rng, DIM * kv_dim)  # wv
        w.raw(floats(rng, DIM, 0.8, 1.2))  # attn sub-norm
        w.align64()
        projection(w, rng, DIM * DIM)  # wo
        projection(w, rng, DIM * HIDDEN)  # w1
        projection(w, rng, DIM * HIDDEN)  # w3
        w.raw(floats(rng, HIDDEN, 0.8, 1.2))  # ffn sub-norm
        w.align64()
        projection(w, rng, DIM * HIDDEN)  # w2
    w.raw(floats(rng, DIM, 0.8, 1.2))  # final norm
    w.align64()
    header = struct.pack("<II", TN_MAGIC, TN_VERSION)
    # dim hidden layers heads kv_heads vocab seq act_type rope_theta scale_mode (40 bytes)
    header += struct.pack("<7i", DIM, HIDDEN, LAYERS, HEADS, KV_HEADS, VOCAB, SEQ)
    header += struct.pack("<i", 1)  # act_type: ReLU2
    header += struct.pack("<f", 10000.0)
    header += struct.pack("<i", 0)  # scale_mode: matrix-scaled ternary with sub-norms
    return header.ljust(64, b"\0") + bytes(w.buf)


def build_tokenizer():
    """vocab_size, max_token_len, then (score float, length int, bytes) per token."""
    tokens = [b"<unk>", b"<s>", b"</s>"] + [bytes([c]) for c in range(32, 127)]
    tokens += [b"t%d" % i for i in range(VOCAB - len(tokens))]
    out = struct.pack("<ii", len(tokens), max(len(t) for t in tokens))
    for i, t in enumerate(tokens):
        out += struct.pack("<fi", float(-i), len(t)) + t
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    rng = random.Random(20261122)
    with open(os.path.join(out, "model.bin"), "wb") as f:
        f.write(build_model(rng))
    with open(os.path.join(out, "tokenizer.bin"), "wb") as f:
        f.write(build_tokenizer())


if __name__ == "__main__":
    main()
