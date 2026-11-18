#!/bin/sh
# check_dist_isa.sh <pz binary> -- the portable (`make dist`) binary must actually contain the
# AVX-512 / VBMI tier code, and VBMI instructions must exist only in code that is bound on a CPU
# with execution-verified VBMI.
#
# Why (docs/ai/mistakes.md, 2026-11-18): the portable build compiles most files at the x86-64-v2 /
# AVX2 baseline. A tier hidden behind a compile-time guard in such a file silently vanishes, or turns
# into a scalar fallback under an AVX-512 name, and the release binary decodes BitNet 2B 35% slower
# than a native build with every test green. Nothing else notices, so this inspects the binary.
#
# Needs: nm, objdump (binutils). Exit 0 = ok, 1 = a check failed, 2 = tool/binary problem.
set -u
BIN=${1:-./pz}
[ -x "$BIN" ] || { echo "check_dist_isa: $BIN is not an executable" >&2; exit 2; }
command -v objdump >/dev/null 2>&1 || { echo "check_dist_isa: objdump not found" >&2; exit 2; }
command -v nm >/dev/null 2>&1 || { echo "check_dist_isa: nm not found" >&2; exit 2; }
case "$(uname -m)" in x86_64|amd64) ;; *) echo "check_dist_isa: skipped on $(uname -m) (x86 only)"; exit 0;; esac

fail=0
DIS=$(mktemp) || exit 2
trap 'rm -f "$DIS"' EXIT
objdump -d --no-show-raw-insn "$BIN" > "$DIS" 2>/dev/null || { echo "check_dist_isa: objdump failed" >&2; exit 2; }

# count INSN-regex matches inside function $1
count_in() {
    awk -v fn="$1" -v re="$2" '
        $0 ~ "^[0-9a-f]+ <" fn ">:" { on = 1; next }
        /^[0-9a-f]+ <.*>:$/ { on = 0 }
        on && $0 ~ re { n++ }
        END { print n + 0 }' "$DIS"
}

# 1. tier code is really vector code in this binary
need() {  # function, instruction regex, label
    n=$(count_in "$1" "$2")
    if [ "$n" -gt 0 ]; then printf '  ok    %-38s %s (%s)\n' "$1" "$3" "$n"
    else printf '  FAIL  %-38s no %s: the tier is missing or a scalar fallback\n' "$1" "$3"; fail=1; fi
}
echo "check_dist_isa: $BIN"
need quantize_row_to_i8_avx512                zmm             "zmm instructions"
need sum_i8_avx512                            zmm             "zmm instructions"
need tn_cls_i4_rows_avx512                    zmm             "zmm instructions"
need tn_cls_i8_rows_avx512                    zmm             "zmm instructions"
need ternary_matmul_packed_vnni               vpdpbusd        "vpdpbusd"
need ternary_matmul_packed_vnni_preq          vpdpbusd        "vpdpbusd"
need ternary_matmul_packed_vnni_vbmi          vpmultishiftqb  "vpmultishiftqb"
need ternary_matmul_packed_vnni_preq_vbmi     vpmultishiftqb  "vpmultishiftqb"

# 2. VBMI instructions only inside *_vbmi functions (and the execution-verification probe), so a
#    VNNI CPU without VBMI (Cascade Lake) never reaches them.
bad=$(awk '
    /^[0-9a-f]+ <.*>:$/ { fn = $2; gsub(/[<>:]/, "", fn); next }
    /vpermb|vpermi2b|vpermt2b|vpmultishiftqb/ && fn !~ /_vbmi$/ && fn != "verify_avx512vbmi_executable" { seen[fn] = 1 }
    END { for (f in seen) print f }' "$DIS")
if [ -n "$bad" ]; then
    echo "  FAIL  VBMI instructions outside *_vbmi functions:"; echo "$bad" | sed 's/^/          /'; fail=1
else
    echo "  ok    VBMI instructions only in *_vbmi functions"
fi

[ "$fail" -eq 0 ] && echo "check_dist_isa: ok" || echo "check_dist_isa: FAILED"
exit "$fail"
