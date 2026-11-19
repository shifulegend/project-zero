#!/bin/sh
# Tests for tools/check_headers_standalone.sh with a throw-away include directory.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
mkdir -p "$T/inc"
pass=0; fail=0
ok()  { pass=$((pass + 1)); echo "  ok   $*"; }
bad() { fail=$((fail + 1)); echo "  FAIL $*"; }

printf '#ifndef TN_A_H\n#define TN_A_H\n#include <stddef.h>\nsize_t f(void);\n#endif\n' > "$T/inc/a.h"
sh "$here/check_headers_standalone.sh" gcc "$T/inc" > "$T/out" 2>&1 && ok "a self-contained header passes" || bad "good header"

printf '#ifndef TN_B_H\n#define TN_B_H\nsize_t g(void);\n#endif\n' > "$T/inc/b.h"
if sh "$here/check_headers_standalone.sh" gcc "$T/inc" > "$T/out" 2>&1; then bad "header missing <stddef.h> rejected"
else grep -q "b.h" "$T/out" && ok "a header that relies on an earlier include is rejected and named" || bad "message names b.h"; fi
echo "=== headers standalone check: $pass passed, $fail failed ==="
[ "$fail" -eq 0 ]
