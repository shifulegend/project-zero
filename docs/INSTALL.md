# Installing Project Zero

The short version is in the [README](../README.md#quick-start). This page has the details.

## Installer (Linux x86-64, no compiler, no root)

```bash
curl -fsSL https://raw.githubusercontent.com/shifulegend/project-zero/master/install.sh | sh
```

It finds the newest release (pre-releases included), checks the archive against the release's
`SHA256SUMS` (and refuses to install on a mismatch or a missing entry), puts `pz` in
`~/.local/bin` (and tells you if that is not on your `PATH`), and offers the optional `pz config`
setup. Options: `--version X.Y.Z`, `--prefix DIR`, `--repo OWNER/NAME`, `--yes`, `--no-wizard`,
`--uninstall` (`sh install.sh --help`). Works from release 0.2.0 on; your settings and models are
never touched.

## By hand

```bash
VER=0.2.0   # example: use the version shown on the Releases page
NAME=project-zero-$VER-x86_64-unknown-linux-gnu
wget https://github.com/shifulegend/project-zero/releases/download/v$VER/$NAME.tar.gz
wget https://github.com/shifulegend/project-zero/releases/download/v$VER/SHA256SUMS
sha256sum -c --ignore-missing SHA256SUMS && tar xf $NAME.tar.gz && cd $NAME
./pz --model /path/to/model.gguf --prompt "What is the capital of France?"
```

The archive holds `pz`, `LICENSE`, `README.md` and `CHANGELOG.md`. Quantized/dense GGUF models need
an AVX2 CPU (Haswell, 2013+); BitNet ternary runs on any x86-64-v2 CPU, and AVX-512/VNNI is used
automatically when present.

## Older releases

v0.1.0 (before the Phase 39 rename) shipped `adaptive_ai_engine-0.1.0-x86_64-linux.tar.gz` with an
`adaptive_ai_engine` binary — same program, old name. v0.2.0 still includes an
`adaptive_ai_engine` → `pz` alias symlink that prints a deprecation notice; it is removed in v0.3.0.

The installer is [`install.sh`](../install.sh); release assets and checksums are described in
[`RELEASING.md`](RELEASING.md#what-gets-shipped).

## Build from source

```bash
git clone https://github.com/shifulegend/project-zero.git
cd project-zero
make demo   # builds engine + downloads SmolLM2-135M + runs a test prompt
```

Expected output: `The capital of France is Paris.` Requirements: GCC or Clang (C and C++
compilers), `make`, and `curl` or `wget`. Build targets are listed in the
[README](../README.md#build).

## Guided setup and persistent settings

`pz config` walks you through five plain-language steps (model,
processor use, memory for long chats, chat-page port, update notices). Each step says what it
means, shows example choices and their effect, and how to change it later. It can download a
checksum-verified starter model suited to your computer's free memory and open the chat page
when you are done. It writes
`~/.config/project-zero/config.toml` (`PZ_CONFIG` overrides the location); `pz config
list|get|set|add|unset|path` edits it non-interactively, and every key can also be set as a
`PZ_<KEY>` environment variable. Precedence: command-line flag > environment > file > default;
`--no-config` ignores the file and environment.

Implementation: [`src/cli/config_cmd.c`](../src/cli/config_cmd.c) (guided setup),
[`src/config/pz_config.c`](../src/config/pz_config.c) (keys, validation, file format).
