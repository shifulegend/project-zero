#!/bin/sh
# Tests for tools/check_root_files.sh with fixture file lists.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
pass=0; fail=0
ok()  { pass=$((pass + 1)); echo "  ok   $*"; }
bad() { fail=$((fail + 1)); echo "  FAIL $*"; }

printf 'README.md\nMakefile\nsrc/core/a.c\ntests/fixtures/x.jpg\n' > "$T/good"
sh "$here/check_root_files.sh" "$T/good" >/dev/null 2>&1 && ok "allowed root files and any subdirectory file pass" || bad "allowed list"

printf 'README.md\nQUICK_SUMMARY.txt\n' > "$T/scratch"
if sh "$here/check_root_files.sh" "$T/scratch" > "$T/out" 2>&1; then bad "scratch .txt at root rejected"
else grep -q "QUICK_SUMMARY.txt" "$T/out" && ok "a stray .txt at the root is rejected and named" || bad "message names the file"; fi

printf 'strawberry.jpg\nresults.csv\n' > "$T/two"
if sh "$here/check_root_files.sh" "$T/two" > "$T/out" 2>&1; then bad "several strays rejected"
else [ "$(grep -c 'unexpected file' "$T/out")" -eq 2 ] && ok "every stray is reported" || bad "all strays reported"; fi

sh "$here/check_root_files.sh" >/dev/null 2>&1 && ok "the real tracked tree passes" || bad "real tree passes"
echo "=== root files check: $pass passed, $fail failed ==="
[ "$fail" -eq 0 ]
