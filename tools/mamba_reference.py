#!/usr/bin/env python3
# mamba_reference.py — independent numpy Mamba-1 reference for Phase 22-SSM.
#
# Reads an F16/F32 Mamba GGUF directly (no llama.cpp, no torch), runs the
# same token ids through a straightforward Mamba-1 implementation and prints
# the top-5 logits for each greedy step, for comparison with
#   pz --model <m> --prompt ... --temperature 0 --verbose
# (its "[DBG] step=N ... top5:[...]" lines).
#
# Usage: tools/mamba_reference.py <model.gguf> <comma-separated token ids> <n_gen>
# Tooling only (numpy); not needed to build or run the engine.
import struct
import sys
from typing import Any

import numpy as np
import numpy.typing as npt

path = sys.argv[1]
toks = [int(t) for t in sys.argv[2].split(",")]
n_gen = int(sys.argv[3])
with open(path, "rb") as f:
    data = f.read()
off = 0


def rd(fmt: str) -> Any:
    """Reads one value of struct format `fmt` from the GGUF data at the global offset and advances it."""
    global off
    v = struct.unpack_from(fmt, data, off)
    off += struct.calcsize(fmt)
    return v[0]


def rs() -> str:
    """Reads a length-prefixed string."""
    n = rd("<Q")
    global off
    s = data[off : off + n].decode()
    off += n
    return s


T = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}


def rv(t: int) -> Any:
    """Reads a metadata value of GGUF type `t`."""
    global off
    if t in T:
        return rd(T[t])
    if t == 8:
        return rs()
    et = rd("<I")
    n = rd("<Q")
    return [rv(et) for _ in range(n)]


assert data[:4] == b"GGUF"
off = 4
ver = rd("<I")
nt = rd("<Q")
nk = rd("<Q")
meta = {}
for _ in range(nk):
    k = rs()
    t = rd("<I")
    meta[k] = rv(t)
align = meta.get("general.alignment", 32)
tensors = {}
for _ in range(nt):
    name = rs()
    nd = rd("<I")
    dims = [rd("<Q") for _ in range(nd)]
    typ = rd("<I")
    o = rd("<Q")
    tensors[name] = (dims, typ, o)
base = (off + align - 1) // align * align


def get(name: str) -> npt.NDArray[np.float32]:
    """The named tensor as a float array."""
    dims, typ, o = tensors[name]
    n = int(np.prod(dims))
    dt = {0: np.float32, 1: np.float16}[typ]
    a = np.frombuffer(data, dtype=dt, count=n, offset=base + o).astype(np.float32)
    return a.reshape(list(reversed(dims)))  # numpy shape = (ne1, ne0)


L = meta["mamba.block_count"]
di = meta["mamba.ssm.inner_size"]
ds = meta["mamba.ssm.state_size"]
dc = meta["mamba.ssm.conv_kernel"]
dr = meta["mamba.ssm.time_step_rank"]
eps = meta["mamba.attention.layer_norm_rms_epsilon"]
emb = get("token_embd.weight")  # (vocab, dim)
W = [
    {
        k: get(f"blk.{l}.{k}")
        for k in [
            "attn_norm.weight",
            "ssm_in.weight",
            "ssm_conv1d.weight",
            "ssm_conv1d.bias",
            "ssm_x.weight",
            "ssm_dt.weight",
            "ssm_dt.bias",
            "ssm_a",
            "ssm_d",
            "ssm_out.weight",
        ]
    }
    for l in range(L)
]
onorm = get("output_norm.weight")
conv = [np.zeros((di, dc - 1), np.float32) for _ in range(L)]
ssm = [np.zeros((di, ds), np.float32) for _ in range(L)]


def rms(x: npt.NDArray[np.float32], w: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """RMS norm of `x` with weight `w`."""
    return np.asarray(x / np.sqrt(np.mean(x * x) + eps) * w)


def silu(v: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Sigmoid linear unit."""
    return np.asarray(v / (1 + np.exp(-v)))


def softplus(v: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """log(1 + e^v), linear above 20 where the two agree to float precision."""
    return np.asarray(np.where(v > 20, v, np.log1p(np.exp(v))))


def step(tok: int) -> npt.NDArray[np.float32]:
    """Runs one token through every Mamba layer and returns the output logits."""
    x = emb[tok].copy()
    for l in range(L):
        w = W[l]
        h = rms(x, w["attn_norm.weight"])
        xz = w["ssm_in.weight"] @ h
        u, z = xz[:di], xz[di:]
        win = np.concatenate([conv[l], u[:, None]], axis=1)  # (di, dc)
        uc = silu((win * w["ssm_conv1d.weight"]).sum(1) + w["ssm_conv1d.bias"])
        conv[l] = win[:, 1:]
        xdb = w["ssm_x.weight"] @ uc
        dtr, B, C = xdb[:dr], xdb[dr : dr + ds], xdb[dr + ds :]
        dt = softplus(w["ssm_dt.weight"] @ dtr + w["ssm_dt.bias"])
        ssm[l] = ssm[l] * np.exp(dt[:, None] * w["ssm_a"]) + (dt * uc)[:, None] * B[None, :]
        y = ssm[l] @ C + w["ssm_d"] * uc
        x = x + w["ssm_out.weight"] @ (y * silu(z))
    return emb @ rms(x, onorm)


seq = list(toks)
for _i, t in enumerate(seq):
    logits = step(t)
for g in range(n_gen):
    top = np.argsort(-logits)[:5]
    print(f"gen{g} top5:", " ".join(f"{i}({logits[i]:.3f})" for i in top))
    nxt = int(top[0])
    logits = step(nxt)
