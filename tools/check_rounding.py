#!/usr/bin/env python3
"""
check_rounding.py -- no truncating float->integer conversions in quantisation code.

Why (docs/ai/mistakes.md, 2026-11-18): the activation quantiser behind every int8 kernel truncated toward
zero (`_mm512_cvttps_epi32`, `(int8_t)(int)v`) while the specification (include/math/quantize_i8.h) and
the reference implementations (bitnet.cpp `roundf`, the BitNet paper) round to nearest. Every tier
truncated the same way, so tier-vs-tier tests agreed, and BitNet perplexity was 9.7 % worse than it should
be. Quantisation must round (cvtps / vcvtn / lrintf / nearbyintf), never truncate.

Flags, in src/ and include/ (third-party headers excluded):
  * truncating x86 converts   _mm*_cvtt*  (cvttps_epi32 ...)
  * truncating NEON converts  vcvtq_s32_f32 / vcvt_s32_f32 and the s8/s16 forms (the rounding ones are vcvtn*, vcvta*)
  * bare truncation casts     (int8_t)(int)v   (tn_i8)(int)v

A line that really needs truncation (an index computation, say) carries `trunc-ok: <reason>` on the same
or the previous line.

Usage: python3 tools/check_rounding.py [--self-test]      (exit 1 on a violation)
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCAN = ["src", "include"]
SKIP = {"stb_image_resize2.h", "stb_image.h", "stb_image_write.h"}
RULES = [
    (re.compile(r"_mm(?:256|512)?_cvtt\w*"), "truncating x86 convert (use _mm*_cvtps_epi32, round to nearest)"),
    (re.compile(r"\bvcvtq?_(?:s|u)(?:8|16|32|64)_f(?:16|32|64)\b"), "truncating NEON convert (use vcvtn*/vcvta*)"),
    (re.compile(r"\((?:int8_t|tn_i8|int16_t)\)\s*\(int\)\s*(?!(?:nearbyint|lrint|rint|round|floor|ceil))[A-Za-z_(]"),
     "bare (int8_t)(int)v truncation cast (use lrintf / nearbyintf)"),
]


def strip_comment(line):
    return re.sub(r"/\*.*?\*/", "", line).split("//")[0]


def scan_text(text):
    """-> [(line_no, message)] for one file's text."""
    lines, out = text.split("\n"), []
    for i, line in enumerate(lines):
        code = strip_comment(line)
        for rx, msg in RULES:
            if rx.search(code) and "trunc-ok" not in line and "trunc-ok" not in (lines[i - 1] if i else ""):
                out.append((i + 1, msg))
    return out


def self_test():
    bad = ["__m512i i = _mm512_cvttps_epi32(v);", "q[i] = (int8_t)(int)v;", "int32x4_t a = vcvtq_s32_f32(f);"]
    good = ["__m512i i = _mm512_cvtps_epi32(v);", "q[i] = (int8_t)lrintf(v);", "int32x4_t a = vcvtnq_s32_f32(f);",
            "/* was _mm512_cvttps_epi32 */ int x = 1;", "int k = (int8_t)(int)v; /* trunc-ok: test */", "int8_t q = (int8_t)(int)nearbyintf(v);"]
    assert all(scan_text(b) for b in bad), "rules must flag the bad samples"
    assert not any(scan_text(g) for g in good), "rules must pass the good samples"
    print("check_rounding self-test ok")


def main():
    if "--self-test" in sys.argv:
        self_test()
        return 0
    bad = 0
    for d in SCAN:
        for p in sorted((ROOT / d).rglob("*")):
            if p.suffix not in (".c", ".h") or p.name in SKIP:
                continue
            for n, msg in scan_text(p.read_text(errors="replace")):
                print(f"{p.relative_to(ROOT)}:{n}: {msg}")
                bad += 1
    print("check_rounding: " + ("FAILED" if bad else "ok"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
