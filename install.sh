#!/bin/sh
# Project Zero installer (Phase 40.4, 2026-10-22).
#
#   curl -fsSL https://raw.githubusercontent.com/shifulegend/project-zero/master/install.sh | sh
#   sh install.sh [--version X.Y.Z] [--prefix DIR] [--yes] [--no-wizard] [--uninstall]
#
# Downloads the release archive for this machine, verifies it against the
# release's SHA256SUMS (refuses to install on a mismatch or a missing entry),
# and installs `pz` for the current user -- no root needed. Nothing is
# configured here: Project Zero detects the hardware itself. Afterwards it
# offers `pz config` (the optional setup wizard) when run from a terminal.
#
# Options:
#   --version X.Y.Z   install this release (default: the newest, prereleases included)
#   --prefix DIR      install under DIR (default: $HOME/.local; pz goes to DIR/bin)
#   --repo OWNER/NAME release repository (default: shifulegend/project-zero)
#   --yes, -y         no questions (skips the setup wizard offer)
#   --no-wizard       do not offer the setup wizard
#   --uninstall       remove what a previous install put under --prefix
#   --help, -h
# Environment: PZ_INSTALL_PREFIX, PZ_INSTALL_VERSION, PZ_INSTALL_REPO,
#   PZ_INSTALL_BASE_URL (download mirror: <base>/v<version>/<asset>),
#   PZ_INSTALL_API (release-list API base, default https://api.github.com).
# Your settings (~/.config/project-zero/config.toml) are never touched.
set -eu

REPO="${PZ_INSTALL_REPO:-shifulegend/project-zero}"
PREFIX="${PZ_INSTALL_PREFIX:-${HOME:-}/.local}"
VERSION="${PZ_INSTALL_VERSION:-}"
API="${PZ_INSTALL_API:-https://api.github.com}"
BASE_URL="${PZ_INSTALL_BASE_URL:-}"
ASSUME_YES=0
WIZARD=1
UNINSTALL=0

say() { printf '%s\n' "$*"; }
err() { printf 'install.sh: %s\n' "$*" >&2; }
die() { err "$*"; exit 1; }

usage() {
    cat <<'EOF_USAGE'
Usage: install.sh [--version X.Y.Z] [--prefix DIR] [--repo OWNER/NAME] [--yes] [--no-wizard]
       install.sh --uninstall [--prefix DIR]
Installs the Project Zero release for this machine (checksum-verified) into
DIR/bin (default ~/.local/bin). Environment: PZ_INSTALL_PREFIX, PZ_INSTALL_VERSION,
PZ_INSTALL_REPO, PZ_INSTALL_BASE_URL (download mirror), PZ_INSTALL_API.
EOF_USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --version)   [ $# -ge 2 ] || die "--version needs a value"; VERSION="$2"; shift ;;
        --version=*) VERSION="${1#*=}" ;;
        --prefix)    [ $# -ge 2 ] || die "--prefix needs a value"; PREFIX="$2"; shift ;;
        --prefix=*)  PREFIX="${1#*=}" ;;
        --repo)      [ $# -ge 2 ] || die "--repo needs a value"; REPO="$2"; shift ;;
        --repo=*)    REPO="${1#*=}" ;;
        -y|--yes)    ASSUME_YES=1 ;;
        --no-wizard) WIZARD=0 ;;
        --uninstall) UNINSTALL=1 ;;
        -h|--help)   usage; exit 0 ;;
        *)           die "unknown option: $1 (see --help)" ;;
    esac
    shift
done

[ -n "$PREFIX" ] || die "no install location: set HOME or pass --prefix"
PREFIX_ARG=""
[ "$PREFIX" = "${HOME:-}/.local" ] || PREFIX_ARG=" --prefix $PREFIX"
VERSION="${VERSION#v}"
BIN_DIR="$PREFIX/bin"
DOC_DIR="$PREFIX/share/doc/project-zero"
MANIFEST="$PREFIX/share/project-zero/install-manifest"

# ── uninstall ────────────────────────────────────────────────────────────
if [ "$UNINSTALL" -eq 1 ]; then
    [ -f "$MANIFEST" ] || die "nothing to uninstall under $PREFIX (no $MANIFEST)"
    while IFS= read -r f; do
        case "$f" in "$PREFIX"/*) rm -f "$f" ;; esac   # only ever inside the prefix
    done < "$MANIFEST"
    rm -f "$MANIFEST"
    rmdir "$DOC_DIR" "$PREFIX/share/project-zero" 2>/dev/null || true
    say "Removed Project Zero from $PREFIX."
    say "Your settings and models were left in place (see: pz config path)."
    exit 0
fi

# ── platform → release target triple ─────────────────────────────────────
OS="${PZ_INSTALL_OS:-$(uname -s)}"
ARCH="${PZ_INSTALL_ARCH:-$(uname -m)}"
case "$OS/$ARCH" in
    Linux/x86_64|Linux/amd64) TARGET="x86_64-unknown-linux-gnu" ;;
    *)
        err "no pre-built release for $OS/$ARCH yet (Linux x86-64 only)."
        err "Build from source instead (needs a C compiler and make):"
        err "  git clone https://github.com/$REPO.git && cd project-zero && make release"
        exit 2 ;;
esac

# ── tools ────────────────────────────────────────────────────────────────
if command -v curl >/dev/null 2>&1; then
    fetch() { curl -fsSL --retry 3 -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then
    fetch() { wget -q -O "$2" "$1"; }
else
    die "need curl or wget to download"
fi
if command -v sha256sum >/dev/null 2>&1; then
    sha256() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
    sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
    die "need sha256sum or shasum to verify the download"
fi
command -v tar >/dev/null 2>&1 || die "need tar"

TMP="$(mktemp -d 2>/dev/null || mktemp -d -t pz-install)"
trap 'rm -rf "$TMP"' EXIT
trap 'exit 130' INT TERM

# ── version ──────────────────────────────────────────────────────────────
# Every release so far is marked "prerelease", which GitHub's /releases/latest
# skips, so take the newest entry of the release list instead.
if [ -z "$VERSION" ]; then
    fetch "$API/repos/$REPO/releases" "$TMP/releases.json" ||
        die "cannot list releases of $REPO (offline? pass --version X.Y.Z)"
    VERSION="$(sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"v\{0,1\}\([^"]*\)".*/\1/p' "$TMP/releases.json" | head -n 1)"
    [ -n "$VERSION" ] || die "no releases found for $REPO"
fi
case "$VERSION" in
    *[!A-Za-z0-9._+-]*) die "invalid version \"$VERSION\"" ;;
esac

NAME="project-zero-$VERSION-$TARGET"
URL_BASE="${BASE_URL:-https://github.com/$REPO/releases/download}/v$VERSION"

# ── download + verify ────────────────────────────────────────────────────
say "Installing Project Zero $VERSION ($TARGET) into $PREFIX"
if ! fetch "$URL_BASE/$NAME.tar.gz" "$TMP/$NAME.tar.gz"; then
    err "download failed: $URL_BASE/$NAME.tar.gz"
    # Releases before the Phase 39 rename (0.1.x) ship adaptive_ai_engine-<v>-x86_64-linux.tar.gz.
    case "$VERSION" in
        0.0.*|0.1.*) err "v$VERSION predates this installer (old archive name); install 0.2.0 or later," \
                         "or download it by hand from https://github.com/$REPO/releases" ;;
        *) err "check that release v$VERSION exists: https://github.com/$REPO/releases" ;;
    esac
    exit 1
fi
fetch "$URL_BASE/SHA256SUMS" "$TMP/SHA256SUMS" ||
    die "download failed: $URL_BASE/SHA256SUMS -- refusing to install unverified"
WANT="$(awk -v f="$NAME.tar.gz" '$2 == f || $2 == "*" f { print $1; exit }' "$TMP/SHA256SUMS")"
[ -n "$WANT" ] || die "SHA256SUMS has no entry for $NAME.tar.gz -- refusing to install"
GOT="$(sha256 "$TMP/$NAME.tar.gz")"
[ "$GOT" = "$WANT" ] || die "checksum mismatch for $NAME.tar.gz (expected $WANT, got $GOT) -- refusing to install"
say "Checksum verified."

mkdir "$TMP/x"
tar -xzf "$TMP/$NAME.tar.gz" -C "$TMP/x"
SRC="$TMP/x/$NAME"
if [ ! -f "$SRC/pz" ] || [ -L "$SRC/pz" ]; then die "archive does not contain $NAME/pz"; fi

# ── install ──────────────────────────────────────────────────────────────
mkdir -p "$BIN_DIR" "$DOC_DIR" "$(dirname "$MANIFEST")"
cp "$SRC/pz" "$BIN_DIR/.pz.new"
chmod 0755 "$BIN_DIR/.pz.new"
mv -f "$BIN_DIR/.pz.new" "$BIN_DIR/pz"          # atomic: a running pz keeps its old file
printf '%s\n' "$BIN_DIR/pz" > "$MANIFEST.new"
for f in LICENSE README.md CHANGELOG.md; do
    if [ -f "$SRC/$f" ]; then
        cp "$SRC/$f" "$DOC_DIR/$f"
        printf '%s\n' "$DOC_DIR/$f" >> "$MANIFEST.new"
    fi
done
mv -f "$MANIFEST.new" "$MANIFEST"

"$BIN_DIR/pz" --version >/dev/null 2>&1 || die "installed $BIN_DIR/pz does not run on this machine"
say "Installed $("$BIN_DIR/pz" --version 2>/dev/null | head -n 1) to $BIN_DIR/pz"

case ":${PATH:-}:" in
    *":$BIN_DIR:"*) ;;
    *) say ""
       say "$BIN_DIR is not on your PATH. Add it, e.g.:"
       say "  echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.profile && . ~/.profile" ;;
esac

say ""
say "Next: get a model (any GGUF file, e.g. from huggingface.co) and run"
say "  pz --model <file.gguf> --prompt \"Hello\"      or      pz --model <file.gguf> --server"
say "Uninstall: sh install.sh --uninstall$PREFIX_ARG"

# ── optional setup wizard (needs a terminal; works under `curl | sh` too) ─
if [ "$WIZARD" -eq 1 ] && [ "$ASSUME_YES" -eq 0 ] && [ -r /dev/tty ] && [ -w /dev/tty ] &&
   { : < /dev/tty; } 2>/dev/null; then
    printf '\nRun the guided setup now? It explains each choice in plain words, can download\n'  > /dev/tty
    printf 'a starter model for you and open the chat page. (About a minute.) [Y/n] ' > /dev/tty
    read -r ans < /dev/tty || ans=n
    case "$ans" in
        ''|y|Y|yes|YES) "$BIN_DIR/pz" config < /dev/tty > /dev/tty 2>&1 || true ;;
        *) say "Skipped. Run \`pz config\` any time." ;;
    esac
fi
