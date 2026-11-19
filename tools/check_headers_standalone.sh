#!/bin/sh
# Every header under include/ (except vendored stb_*) must compile on its own, so that including it
# first in a .c file never fails because it relies on an earlier include.
#
# usage: tools/check_headers_standalone.sh [CC] [DIR]    (default: gcc, include)
set -u
cc="${1:-gcc}"
dir="${2:-include}"
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
bad=0
for h in $(find "$dir" -name '*.h' ! -name 'stb_*' | sort); do
    if ! "$cc" -std=c99 -Wall -Wextra -Wpedantic -Iinclude -I"$dir" -D_POSIX_C_SOURCE=200809L -D_GNU_SOURCE \
         -march=native -fsyntax-only -x c "$h" 2> /tmp/hdr.$$ ; then
        echo "not self-contained: $h"; head -2 /tmp/hdr.$$ | sed 's/^/    /'; bad=$((bad + 1))
    fi
done
rm -f /tmp/hdr.$$
if [ "$bad" -ne 0 ]; then echo "$bad header(s) do not compile on their own"; exit 1; fi
echo "all headers are self-contained ($cc)"
