"""Independent numpy check of BitNet b1.58 2B layer 0 for the BOS token (position 0), straight from the GGUF."""

import sys
from typing import Any

sys.path.insert(0, ".")
import os

import numpy as np
import numpy.typing as npt
from gguf_min import G

g = G(os.environ.get("BITNET_GGUF", "models/bitnet-b1.58-2B-4T-i2_s.gguf"))
EPS = g.kv["bitnet-b1.58.attention.layer_norm_rms_epsilon"]


def f32(name: str) -> npt.NDArray[np.float64]:
    """The named F32 tensor as a float64 vector."""
    d, ty, off = g.t[name]
    n = int(np.prod(d))
    assert ty == 0
    return np.frombuffer(g.raw(name, n * 4), dtype="<f4").astype(np.float64)


def embd(tok: int) -> npt.NDArray[np.float64]:
    """The F16 embedding row of token `tok` as a float64 vector."""
    d, ty, off = g.t["token_embd.weight"]
    assert ty == 1
    row = np.frombuffer(g.raw("token_embd.weight", d[0] * d[1] * 2), dtype="<f2").reshape(
        d[1], d[0]
    )[tok]
    return np.asarray(row, dtype=np.float64)


def rms(x: npt.NDArray[Any], w: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """RMS norm of `x` with weight `w`."""
    return np.asarray(x / np.sqrt(np.mean(x * x) + EPS) * w)


def i2s_matrix(name: str, layout: str = "A") -> tuple[npt.NDArray[np.float32], float]:
    """-> ternary matrix W[out, in] in {-1,0,1} (float32) and the per-tensor scale."""
    (nin, nout), ty, off = g.t[name]
    assert ty == 36
    nb = nin * nout // 4
    raw = np.frombuffer(g.raw(name, nb + 4), dtype=np.uint8)
    scale = np.frombuffer(raw[nb : nb + 4].tobytes(), dtype="<f4")[0]
    packed = raw[:nb]
    if (
        layout == "A"
    ):  # ACT_PARALLEL quantiser: flat 128-element blocks, 32 bytes each, shifts 6,4,2,0 per group of 32
        blk = packed.reshape(-1, 32)  # [nblocks, 32]
        vals = np.stack([(blk >> s) & 3 for s in (6, 4, 2, 0)], axis=1)  # [nblocks, 4(group), 32]
        flat = vals.reshape(-1)  # element order j = group*32 + pos
        W = flat.reshape(nout, nin).astype(np.float32) - 1.0
    else:  # 1x4 quantiser: 4 rows interleaved per column byte (q0<<6 | q1<<4 | q2<<2 | q3)
        p = packed.reshape(nout // 4, nin)
        W = (
            np.stack([(p >> 6) & 3, (p >> 4) & 3, (p >> 2) & 3, p & 3], axis=1)
            .reshape(nout, nin)
            .astype(np.float32)
            - 1.0
        )
    return W, float(scale)


def bitlinear(
    x: npt.NDArray[Any], name: str, layout: str = "A", actq: bool = True
) -> npt.NDArray[Any]:
    """BitNet linear layer: ternary I2_S matrix times `x` (activation-quantised to int8 when `actq`), scaled."""
    W, scale = i2s_matrix(name, layout)
    if actq:
        s = 127.0 / np.max(np.abs(x))
        q = np.clip(np.round(x * s), -128, 127)
        return np.asarray((W.astype(np.float64) @ q) / s * scale)
    return np.asarray((W.astype(np.float64) @ x) * scale)


def show(tag: str, v: npt.NDArray[Any]) -> None:
    """Prints a tag, the first three values and the RMS of a vector."""
    print(
        f"{tag:<26}",
        np.array2string(v[:3], precision=4, floatmode="fixed"),
        f" scale-free rms={np.sqrt(np.mean(v * v)):.4f}",
    )


if __name__ == "__main__":
    layout = sys.argv[1] if len(sys.argv) > 1 else "A"
    x0 = embd(128000)
    show("embd[BOS]", x0)
    n = rms(x0, f32("blk.0.attn_norm.weight"))
    show("attn_norm-0", n)
    V = bitlinear(n, "blk.0.attn_v.weight", layout)
    show("Vcur-0 (V)", V)
    a = np.concatenate([V[(h // 4) * 128 : (h // 4 + 1) * 128] for h in range(20)])
    s = rms(a, f32("blk.0.attn_sub_norm.weight"))
    show("attn_sub_norm-0", s)
    ao = bitlinear(s, "blk.0.attn_output.weight", layout)
    show("attn_out-0", ao)
    fi = x0 + ao
    show("ffn_inp-0", fi)
    fn = rms(fi, f32("blk.0.ffn_norm.weight"))
    show("ffn_norm-0 (dense_ffn_norm)", fn)
    gate = bitlinear(fn, "blk.0.ffn_gate.weight", layout)
    up = bitlinear(fn, "blk.0.ffn_up.weight", layout)
    show("ffn_gate-0", gate)
    show("ffn_up-0", up)
    h = np.maximum(gate, 0) ** 2 * up
    hn = rms(h, f32("blk.0.ffn_sub_norm.weight"))
    dn = bitlinear(hn, "blk.0.ffn_down.weight", layout)
    show("ffn_down-0", dn)
    show("l_out-0", fi + dn)
