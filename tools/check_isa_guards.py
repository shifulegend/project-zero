#!/usr/bin/env python3
"""
check_isa_guards.py -- no AVX-512/VNNI code or compile-time guards in files that are not built
with those ISA flags.

Why (docs/ai/mistakes.md, 2026-11-18): the portable `make dist` binary compiles most files at the
x86-64-v2 / AVX2 baseline and gives only the dedicated kernel files their own ISA flags. A
`#if TN_HAS_AVX512...` section inside a baseline file therefore silently vanishes from the release
binary (or degrades to a scalar fallback under an AVX-512 name), while `make release`
(-march=native) keeps it. The release binary then runs the same model 20-35% slower than a native
build and nothing fails. Runtime dispatch only works if every tier is COMPILED in every build:

  * tier code lives in a file that carries its ISA flags (Makefile AVX512_TUS / per-file rules,
    CMake PZ_AVX512_SRCS / set_source_files_properties);
  * it is referenced only from simd_dispatch.c (built with TN_FORCE_DISPATCH_ALL) through a
    dispatch pointer, never behind a compile-time macro in a baseline file.

This script finds files that mention AVX-512/VNNI intrinsics or macros but are not built with
those flags. PENDING lists the known offenders still to be split (it must only shrink); a file
that is clean but still listed fails the check, so the list stays honest.

Usage: python3 tools/check_isa_guards.py            (exit 1 on a violation)
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
USE = re.compile(r"_mm512_|__AVX512|TN_HAS_AVX512|TN_HAS_AVXVNNI|__AVXVNNI__")

# Files that legitimately reference the macros without being AVX-512 kernels.
EXEMPT = {
    "src/math/simd_dispatch.c": "built with TN_FORCE_DISPATCH_ALL; the only file that references every tier",
    "src/runtime/calibration.c": "built with TN_FORCE_DISPATCH_ALL; decides which tiers exist to benchmark",
    "src/core/cpu_features.c": "CPUID probe and execution-verification of VBMI/VNNI, no tier kernels",
}

# Known offenders still to be split into ISA-flagged files. Shrink only.
PENDING = {
    "src/math/parallel_matmul.c": "AVX-512 BF16 LM-head and Q8/Q4 dequant matvec sections behind TN_HAS_AVX512",
    "src/math/matmul_q5_1.c": "mentions TN_HAS_AVX512 in a guard only (no AVX-512 code): simplify with the rest",
    "src/math/matmul_f16.c": "AVX-512 F16 matvec behind TN_HAS_AVX512 in an AVX2-flagged file",
    "src/math/batched_matmul_f16.c": "AVX-512 F16 convert behind TN_HAS_AVX512",
    "src/math/matmul_q8_0.c": "AVX-512 section behind TN_HAS_AVX512",
    "src/math/matmul_q5k.c": "AVX-512 section behind TN_HAS_AVX512",
    "src/math/matmul_q5_0.c": "AVX-512 section behind TN_HAS_AVX512",
    "src/math/matmul_q4k.c": "AVX2+VNNI dot path behind TN_HAS_AVX512VNNI",
    "src/math/q8_0_act.c": "AVX2+VNNI path behind TN_HAS_AVX512VNNI",
    "src/math/matmul_q2_0.c": "VNNI dispatch behind TN_HAS_AVX512VNNI (kernel is in matmul_q2_0_vnni.c)",
    "src/transformer/attention.c": "AVX-512 section in a baseline file",
    "src/sampling/temperature.c": "AVX-512 section in a baseline file",
}


def isa_flagged_files():
    """Files the Makefile builds with AVX-512 / VNNI per-file flags."""
    mk = (ROOT / "Makefile").read_text()
    flagged = set()
    m = re.search(r"^AVX512_TUS\s*:=\s*((?:.*\\\n)*.*)$", mk, re.M)
    if m:
        for t in re.findall(r"[A-Za-z0-9_]+/[A-Za-z0-9_]+", m.group(1).replace("\\\n", " ")):
            flagged.add(f"src/{t}.c")
    # explicit per-file rules: "build/<x>.o: src/<x>.c" plus its whole tab-indented recipe
    for rule in re.finditer(r"^build/([^\s:]+)\.o:[^\n]*\n((?:\t[^\n]*\n?)+)", mk, re.M):
        if re.search(r"-mavx512|-mavxvnni", rule.group(2)):
            flagged.add(f"src/{rule.group(1)}.c")
    return flagged


def strip_comments(text):
    """Drop /* */ and // comments: prose about a macro is not a use of it."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def main():
    """Entry point: checks that AVX-512/VNNI code only appears in files built with those ISA flags (and that the pending list has no stale entries); returns the exit code."""
    flagged = isa_flagged_files()
    if not flagged:
        print("check_isa_guards: could not read the ISA file lists from the Makefile")
        return 2
    bad, stale = [], []
    for path in sorted((ROOT / "src").rglob("*.c")):
        rel = path.relative_to(ROOT).as_posix()
        uses = bool(USE.search(strip_comments(path.read_text(errors="replace"))))
        if rel in flagged or rel in EXEMPT:
            continue
        if uses and rel not in PENDING:
            bad.append(rel)
        if not uses and rel in PENDING:
            stale.append(rel)
    ok = True
    if bad:
        ok = False
        print("AVX-512/VNNI code or guards in files built without those ISA flags:")
        for f in bad:
            print(f"  {f}")
        print(
            "Move the tier into an ISA-flagged file and bind it from simd_dispatch.c "
            "(see the header of this script)."
        )
    if stale:
        ok = False
        print("Listed in PENDING but clean now -- remove from tools/check_isa_guards.py:")
        for f in stale:
            print(f"  {f}")
    print(
        f"check_isa_guards: {'ok' if ok else 'FAILED'} "
        f"({len(flagged)} ISA-flagged files, {len(PENDING)} pending split)"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
