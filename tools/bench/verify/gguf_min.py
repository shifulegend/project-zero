"""Minimal GGUF reader (header, kv, tensor infos, mmap) for the numpy reference. No pz or llama.cpp code used."""

import mmap
import struct
from typing import Any


class G:
    """G."""

    def __init__(s, path: str) -> None:
        """Set up the instance."""
        s.f = open(path, "rb")
        s.m = mmap.mmap(s.f.fileno(), 0, access=mmap.ACCESS_READ)
        b = s.m
        s.o = 0
        assert b[:4] == b"GGUF"
        s.o = 4
        (s.ver,) = s.u("I")
        (s.nt,) = s.u("Q")
        (s.nkv,) = s.u("Q")
        s.kv: dict[str, Any] = {}  # metadata values are heterogeneous (scalars, strings, lists)
        for _ in range(s.nkv):
            k = s.s()
            (t,) = s.u("I")
            s.kv[k] = s.val(t)
        s.t: dict[str, tuple[list[int], int, int]] = {}  # name -> (dims, type, offset)
        for _ in range(s.nt):
            n = s.s()
            (nd,) = s.u("I")
            dims = [s.u("Q")[0] for _ in range(nd)]
            (ty,) = s.u("I")
            (off,) = s.u("Q")
            s.t[n] = (dims, ty, off)
        s.align = s.kv.get("general.alignment", 32)
        s.data0 = (s.o + s.align - 1) // s.align * s.align

    def u(s, fmt: str) -> tuple[Any, ...]:
        """Reads one little-endian value of struct format `fmt` and advances the offset."""
        sz = struct.calcsize("<" + fmt)
        r = struct.unpack_from("<" + fmt, s.m, s.o)
        s.o += sz
        return r

    def s(s_) -> str:
        """Reads a length-prefixed UTF-8 string and advances the offset."""
        (n,) = s_.u("Q")
        v = bytes(s_.m[s_.o : s_.o + n]).decode("utf8", "replace")
        s_.o += n
        return v

    def val(s, t: int) -> Any:
        """Reads a metadata value of GGUF type `t` (scalar, string or array)."""
        if t == 8:
            return s.s()
        if t == 9:
            (et,) = s.u("I")
            (c,) = s.u("Q")
            if et == 8 or c < 100:
                return [s.val(et) for _ in range(c)]
            s.skip(et, c)
            return "[array]"
        fm = {
            0: "B",
            1: "b",
            2: "H",
            3: "h",
            4: "I",
            5: "i",
            6: "f",
            7: "?",
            10: "Q",
            11: "q",
            12: "d",
        }[t]
        return s.u(fm)[0]

    def skip(s, et: int, c: int) -> None:
        """Skips `c` array elements of GGUF type `et`."""
        if et == 8:
            for _ in range(c):
                s.s()
        else:
            s.o += c * struct.calcsize(
                "<"
                + {
                    0: "B",
                    1: "b",
                    2: "H",
                    3: "h",
                    4: "I",
                    5: "i",
                    6: "f",
                    7: "?",
                    10: "Q",
                    11: "q",
                    12: "d",
                }[et]
            )

    def raw(s, name: str, nbytes: int) -> bytes:
        """The raw bytes of a tensor's data (the first `nbytes`)."""
        dims, ty, off = s.t[name]
        a = s.data0 + off
        return s.m[a : a + nbytes]
