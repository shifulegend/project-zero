#!/bin/sh
# tools/test_install.sh — Phase 40.4 installer test, offline: packages a
# binary the way release.yml does into a file:// mirror (plus a fake release
# list), then checks install, version discovery, checksum refusal, platform
# refusal, upgrade and uninstall -- under dash and bash.
#
# Usage: sh tools/test_install.sh [path/to/pz]   (default ./pz; needs curl)
# shellcheck disable=SC2015  # "cond && ok || bad": ok() always succeeds
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${1:-$ROOT/pz}"
[ -x "$BIN" ] || { echo "no binary at $BIN (run make release)"; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl is required (file:// mirror)"; exit 1; }

W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT
TARGET=x86_64-unknown-linux-gnu
pass=0
fail=0
ok()  { pass=$((pass + 1)); echo "  ok   $*"; }
bad() { fail=$((fail + 1)); echo "  FAIL $*"; }

package() {   # package <version> [corrupt]
    v="$1"; name="project-zero-$v-$TARGET"
    d="$W/mirror/v$v"
    mkdir -p "$d" "$W/pkg/$name"
    cp "$BIN" "$W/pkg/$name/pz"
    ln -sf pz "$W/pkg/$name/adaptive_ai_engine"
    cp "$ROOT/LICENSE" "$ROOT/NOTICE" "$ROOT/README.md" "$ROOT/CHANGELOG.md" "$W/pkg/$name/"
    tar -C "$W/pkg" -czf "$d/$name.tar.gz" "$name"
    (cd "$d" && sha256sum "$name.tar.gz" > SHA256SUMS)
}

package 0.0.1-test
package 0.0.2-test
mkdir -p "$W/api/repos/o/r"
printf '[{"tag_name": "v0.0.2-test", "prerelease": true},\n {"tag_name": "v0.0.1-test"}]\n' \
    > "$W/api/repos/o/r/releases"

export PZ_INSTALL_BASE_URL="file://$W/mirror"
export PZ_INSTALL_API="file://$W/api"
export PZ_INSTALL_REPO=o/r

for SH in dash bash; do
    command -v "$SH" >/dev/null 2>&1 || { echo "($SH not installed, skipped)"; continue; }
    echo "== $SH =="
    P="$W/prefix-$SH"

    if "$SH" "$ROOT/install.sh" --version 0.0.1-test --prefix "$P" --yes > "$W/out" 2>&1 &&
       [ -x "$P/bin/pz" ] && "$P/bin/pz" --version > /dev/null 2>&1; then
        ok "install a given version"
    else bad "install a given version"; cat "$W/out"; fi
    # No curl on PATH: the wget branch must read a file:// mirror itself (wget cannot) and must not
    # be asked for the file:// URL at all.
    NC="$W/nocurl"
    if [ ! -d "$NC" ]; then
        mkdir -p "$NC"
        for f in /usr/bin/* /bin/*; do
            b="${f##*/}"
            [ "$b" = curl ] || [ -e "$NC/$b" ] || ln -s "$f" "$NC/$b"
        done
        rm -f "$NC/wget"
        printf '#!/bin/sh\necho "$@" >> "%s/wget.log"\nexit 1\n' "$W" > "$NC/wget"
        chmod +x "$NC/wget"
    fi
    PW="$W/prefix-wget-$SH"
    if PATH="$NC" "$(command -v "$SH")" "$ROOT/install.sh" --version 0.0.1-test --prefix "$PW" --yes \
        > "$W/out2" 2>&1 && [ -x "$PW/bin/pz" ] && [ ! -e "$W/wget.log" ]; then
        ok "wget fallback installs from a file:// mirror without calling wget"
    else bad "wget fallback installs from a file:// mirror without calling wget"; cat "$W/out2"; fi
    [ -f "$P/share/doc/project-zero/LICENSE" ] && [ -f "$P/share/project-zero/install-manifest" ] &&
        ok "docs + manifest installed" || bad "docs + manifest installed"
    # the notice must be installed byte-identical and recorded in the manifest.
    if cmp -s "$ROOT/NOTICE" "$P/share/doc/project-zero/NOTICE" &&
       grep -q "/NOTICE\$" "$P/share/project-zero/install-manifest"; then
        ok "NOTICE installed and recorded in the manifest"
    else bad "NOTICE installed and recorded in the manifest"; fi
    grep -q "Checksum verified" "$W/out" && ok "checksum reported" || bad "checksum reported"

    if "$SH" "$ROOT/install.sh" --prefix "$P" --yes > "$W/out" 2>&1 &&
       grep -q "Installing Project Zero 0.0.2-test" "$W/out"; then
        ok "newest release (prerelease) discovered and upgraded in place"
    else bad "version discovery / upgrade"; cat "$W/out"; fi

    # tampered archive: the checksum no longer matches
    T="$W/tamper-$SH"; cp -r "$W/mirror" "$T"
    printf 'x' >> "$T/v0.0.1-test/project-zero-0.0.1-test-$TARGET.tar.gz"
    if PZ_INSTALL_BASE_URL="file://$T" "$SH" "$ROOT/install.sh" --version 0.0.1-test \
         --prefix "$W/bad-$SH" --yes > "$W/out" 2>&1; then
        bad "tampered archive refused"
    else
        grep -q "checksum mismatch" "$W/out" && [ ! -e "$W/bad-$SH/bin/pz" ] &&
            ok "tampered archive refused, nothing installed" || { bad "tampered archive message"; cat "$W/out"; }
    fi
    # SHA256SUMS without an entry for the archive
    : > "$T/v0.0.2-test/SHA256SUMS"
    if PZ_INSTALL_BASE_URL="file://$T" "$SH" "$ROOT/install.sh" --version 0.0.2-test \
         --prefix "$W/bad-$SH" --yes > "$W/out" 2>&1; then
        bad "missing checksum entry refused"
    else grep -q "no entry" "$W/out" && ok "missing checksum entry refused" || bad "missing entry message"; fi

    # plain http:// downloads are refused (curl: protocol disabled), nothing installed
    if command -v curl >/dev/null 2>&1; then
        if PZ_INSTALL_BASE_URL="http://127.0.0.1:9" "$SH" "$ROOT/install.sh" --version 0.0.1-test \
             --prefix "$W/http-$SH" --yes > "$W/out" 2>&1; then
            bad "http:// download refused"
        else
            [ ! -e "$W/http-$SH/bin/pz" ] && ok "http:// download refused, nothing installed" || bad "http:// download refused"
        fi
    fi

    if PZ_INSTALL_ARCH=aarch64 "$SH" "$ROOT/install.sh" --prefix "$W/arm-$SH" --yes > "$W/out" 2>&1; then
        bad "unsupported platform refused"
    else
        [ $? -eq 2 ] && grep -q "Build from source" "$W/out" && ok "unsupported platform: exit 2 + source hint" ||
            bad "unsupported platform message"
    fi
    if "$SH" "$ROOT/install.sh" --version '0.0.1;rm -rf /' --prefix "$W/inj-$SH" --yes > "$W/out" 2>&1; then
        bad "invalid version refused"
    else grep -q "invalid version" "$W/out" && ok "invalid version refused" || bad "invalid version message"; fi

    "$SH" "$ROOT/install.sh" --help > "$W/out" 2>&1 && grep -q "Usage:" "$W/out" &&
        ok "--help" || bad "--help"
    "$SH" -s -- --help < "$ROOT/install.sh" > "$W/out" 2>&1 && grep -q "Usage:" "$W/out" &&
        ok "--help when piped (curl | sh)" || bad "--help when piped"

    touch "$P/bin/other-tool"
    if "$SH" "$ROOT/install.sh" --uninstall --prefix "$P" > "$W/out" 2>&1 &&
       [ ! -e "$P/bin/pz" ] && [ ! -e "$P/share/doc/project-zero/LICENSE" ] && [ ! -e "$P/share/doc/project-zero/NOTICE" ] &&
       [ -e "$P/bin/other-tool" ]; then
        ok "uninstall removes only what it installed"
    else bad "uninstall"; cat "$W/out"; fi
    if "$SH" "$ROOT/install.sh" --uninstall --prefix "$P" > "$W/out" 2>&1; then
        bad "second uninstall reports nothing to remove"
    else ok "second uninstall reports nothing to remove"; fi
done

echo "=== installer: $pass passed, $fail failed ==="
[ "$fail" -eq 0 ]
