# Releasing project-zero — Prebuilt x86-64 Linux Binary

How a portable, prebuilt `pz` binary is built, verified, and
published as a GitHub Release. The release pipeline lives in
[`.github/workflows/release.yml`](../.github/workflows/release.yml); the build
is driven by the Makefile `dist` target.

## TL;DR — cut a release

```bash
# 1. Pick the version (SemVer, see below) and move the notes in CHANGELOG.md:
#    "## [Unreleased]"  ->  keep an empty "## [Unreleased]" on top, then
#    "## [0.2.0] - 2026-11-01" with the entries that were under Unreleased,
#    and update the link references at the bottom:
#      [Unreleased]: https://github.com/<owner>/project-zero/compare/v0.2.0...HEAD
#      [0.2.0]: https://github.com/<owner>/project-zero/compare/v0.1.0...v0.2.0
python3 tools/changelog.py check            # format lint (also runs in CI)
python3 tools/changelog.py notes v0.2.0     # preview the GitHub release notes
git commit -am "Release v0.2.0"
# 2. Tag that commit on the default branch and push the tag:
git tag -a v0.2.0 -m "Project Zero v0.2.0"
git push origin v0.2.0   # tag push triggers .github/workflows/release.yml
```

The workflow then: refuses the tag if `CHANGELOG.md` has no section for it, runs the full test suite (gcc + clang), builds the portable
binary (`make dist`), packages it, smoke-tests it, runs it on a clean minimal
container, attempts a real-model golden check, and publishes a **prerelease**
with the `.tar.gz` and `SHA256SUMS` attached and that changelog section as its release notes.

To test the build **without publishing**, run the workflow via
**Actions → Release → Run workflow** (`workflow_dispatch`); the publish job only
runs on a `v*` tag push.

## Versioning and the changelog

- **Versions** follow [Semantic Versioning 2.0](https://semver.org/): `MAJOR.MINOR.PATCH`, tags
  `vX.Y.Z`. While on 0.x, a breaking change bumps MINOR (e.g. the Phase 39 rename → 0.2.0) and
  releases are published as prereleases. The binary embeds `git describe --tags`
  (`pz --version` → `Project Zero v0.2.0`, or `v0.2.0-5-gabc1234` between releases), and the
  update notifier compares it with the latest GitHub release.
- **`CHANGELOG.md`** (repo root) follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/):
  newest first, one `## [Unreleased]` section on top, sections in the order `Changed / Deprecated /
  Removed / Added / Fixed / Security`, written for users (what changed and which flag/API), not a commit
  list. **Every user-facing change adds its line under `[Unreleased]` in the same commit.**
  Breaking changes and deprecations always get an entry, with the removal release named.
- How to write entries (wording, breaking changes, deprecations, security, curation):
  [`docs/ai/changelog-guidance.md`](ai/changelog-guidance.md).
- `tools/changelog.py check` lints the format (CI job `changelog`); `tools/changelog.py notes <tag>`
  is what the release workflow publishes, so the GitHub release page, the update banner's link and
  the web UI's "Release notes" link all show the same curated text.

## What gets shipped

| Asset | Contents |
|-------|----------|
| `project-zero-<version>-x86_64-unknown-linux-gnu.tar.gz` | unpacks to `project-zero-<version>-x86_64-unknown-linux-gnu/`: `pz` + `LICENSE` + `README.md` + `CHANGELOG.md` (+ the `adaptive_ai_engine → pz` alias symlink, first release after the rename only) |
| `install.sh` | the installer (Phase 40.4): picks the asset for the machine, verifies it against `SHA256SUMS`, installs `pz` to `~/.local/bin` |
| `SHA256SUMS` | checksums of the tarball and `install.sh` (`sha256sum -c --ignore-missing SHA256SUMS` to verify what you downloaded) |

`install.sh` finds the newest release through the release list (every release so far is a
prerelease, which `/releases/latest` skips) and expects exactly the asset and directory names
above: **renaming an asset means changing `install.sh` and `tools/test_install.sh` in the same
commit** (engineering-rules DoD #6).

**Naming (Phase 39, 2026-10-21):** `<project>-<version>-<target triple>` — the convention ripgrep
and the Rust/LLVM toolchains use. The version drops the tag's leading `v`; the triple says arch,
vendor and libc (`-gnu`: glibc, with libstdc++/libgcc linked statically), so future aarch64, musl
or macOS builds get their own names without breaking download scripts. Releases up to the rename
were named `adaptive_ai_engine-<version>-x86_64-linux.tar.gz`. **Release notes of the first
release after the rename must call out the new binary and asset names**; remove the alias symlink
(Makefile `LEGACY_TARGET`, release.yml `ln -s`) in the release after that.

## The portable build (`make dist`)

`make release` uses `-march=native` and is **not** distributable — on the
AVX-512-VNNI CI runner it would bake in AVX-512 and `SIGILL` on older CPUs. The
distribution build is therefore separate:

```bash
make dist CC=gcc      # -> ./pz (portable)
```

Design — **per-TU multiversioning** (leverages the existing `simd_dispatch`):

- The bulk compiles at a portable baseline: **`-march=x86-64-v2`** (SSE4.2/POPCNT,
  ~2009+ CPUs). Override with `make dist DIST_MARCH="-march=x86-64-v3"` to raise
  the floor to AVX2.
- Each SIMD kernel TU carries its **own** ISA flag (`-mavx2/-mfma/-mf16c`,
  `-mavx512*`, `-mavx512vnni`, `-mavxvnni`) via the per-file rules in the
  Makefile, so AVX2 / AVX-512 / VNNI are **selected at runtime** by CPUID
  (`tn_simd_init` / `tn_cpu_features_detect`), not at compile time.
- `src/math/simd_dispatch.c` stays at the baseline (no SIMD in the always-run
  startup path) but is compiled with `-DTN_FORCE_DISPATCH_ALL` so every runtime
  branch is present even though the dispatcher TU itself emits no AVX.
- Linking: `-static-libstdc++ -static-libgcc`, so the only runtime deps are
  `libc` / `libm` (verify with `ldd`). Built on `ubuntu-22.04` (glibc 2.35);
  runs on glibc ≥ 2.35.

CMake parity: `cmake -DPZ_DIST=ON` produces the same portable binary (the
Makefile remains canonical for releases).

## Minimum-CPU envelope

| Workload | Floor | Notes |
|----------|-------|-------|
| Start / `--version` / BitNet ternary text-gen | **x86-64-v2** (~2009+) | dispatcher + ternary path degrade to scalar below AVX2 |
| Quantized / dense GGUF models (Q4_K, F16, …) | **AVX2** (Haswell 2013+) | the quant matmul kernels are compile-time AVX2-or-scalar and called directly |
| Best throughput | AVX-512 / **VNNI** auto-detected | e.g. Ice Lake / Tiger Lake / Zen4 / Sapphire Rapids |

AVX2 gives ~15× over scalar for ternary matmul, so the binary keeps full runtime
dispatch — a scalar-only portable build would be unacceptably slow.

## Supported models (documented & verified)

| Model | Format | Source | Golden output (`--temperature 0`) |
|-------|--------|--------|-----------------------------------|
| SmolLM2-135M-Instruct | F16 dense GGUF (271 MB) | `bartowski/SmolLM2-135M-Instruct-GGUF` | "The capital of France is **Paris**." |
| BitNet-b1.58-2B-4T | ternary `.bin` (1.1 GB) | `microsoft/bitnet-b1.58-2B-4T` → `tools/convert_hf_bitnet.py` | "…**Paris**" |
| DeepSeek-V2-Lite-Chat | Q4_K_S / Q2_K GGUF (MoE+MLA) | HuggingFace GGUF | "…**Paris**." / Germany→"**Berlin**" |

Runtime variation flags: `--simd {auto,avx2,avx512f,vnni,scalar}`,
`--classifier {auto,bf16,int8,int4,auto-fast}`, `--threads N`. See `README.md`.

## Verification (run before tagging)

```bash
make test CC=gcc && make test CC=clang        # 46 tests, ASan/UBSan, mock weights
make dist CC=gcc
file ./pz                      # ELF x86-64
ldd  ./pz                      # only libc/libm/ld (no libstdc++/libgcc)
./pz --version                 # prints version + detected backend, exit 0
for b in scalar avx2 avx512f vnni; do TN_FORCE_BACKEND=$b ./pz --version; done

# Real-model golden output (network permitting):
curl -fsSL -o models/smollm2.gguf \
  https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-f16.gguf
./pz --model models/smollm2.gguf \
  --prompt "What is the capital of France?" --max-tokens 30 --temperature 0 --threads 2 | grep -i paris
```

The release workflow runs all of the above; the golden job is `continue-on-error`
because the HuggingFace download is gated by the environment's network policy.

## Versioning

- SemVer; stay on `0.MINOR.PATCH` while pre-1.0 and publish as **prerelease**.
- The tag (`vX.Y.Z`) is the version embedded in the binary (`--version`), via
  `git describe` → `-DPZ_VERSION` (Makefile / CMake).

## Maintaining published releases (standing rules, 2026-10-22)

Sources: SemVer 2.0 §3 ("once a versioned package has been released, the contents of that version
MUST NOT be modified; any modifications MUST be released as a new version"), Keep a Changelog 1.1
(`[YANKED]`), GitHub release immutability, OpenSSF coordinated-vulnerability-disclosure guide.

1. **A published release never changes.** Never replace or delete its assets, never move, delete
   or re-push its tag, never rebuild it. Users' checksums, `install.sh` and reproducibility depend
   on that. Any fix ships as a **new version**.
2. **Only the release notes may be edited**, and only to *add* a dated notice at the top
   (superseded by vX, known issues, security advisory link). Never rewrite or remove the
   original notes.
3. **Every fix that lands is triaged against the releases people use** (the latest release, and
   any release `SECURITY.md` lists as supported):
   - security issue reachable in a shipped feature, data loss, or a crash on supported
     hardware → a release within `SECURITY.md`'s timeline (critical: 14 days). A patch release
     (`x.y.Z`) when the fix is small; when `[Unreleased]` already holds a release's worth of
     tested work, the next minor release *is* the fix — don't hold it back for features;
   - anything else → rolls into the next planned release.
   Record the triage outcome in `docs/ai/decision-log.md`.
4. **Security fixes**: coordinate through a private GitHub Security Advisory (draft → fix →
   publish with the release), request a CVE when the impact warrants it, list them under
   `### Security` in the changelog, credit the reporter, and add a notice to every affected
   release's notes (rule 2).
5. **A broken release** (wrong or unusable artifact, build that crashes on supported CPUs, a
   security hole with no quick fix): publish the fixed version first, then mark the bad one
   `## [x.y.z] - date [YANKED]` in `CHANGELOG.md` and add a notice to its notes. Keep its assets
   unless they are malicious or must go for legal reasons — deleting breaks checksums and history.
6. **Supported versions** are stated in `SECURITY.md` (pre-1.0: the latest release and `master`;
   no backports to older 0.x releases). Change the table when that policy changes.
7. **Before tagging** (in addition to "Verification" above): every workflow green on the tag
   commit — CI (incl. macOS, golden, installer) **and** the Independent Security Audit;
   `CHANGELOG.md` section promoted and `tools/changelog.py check` clean; artifacts built only by
   `release.yml` from the tag.
8. **After publishing**: download the live assets on a clean machine, verify `SHA256SUMS`, run
   `install.sh --version <v>` and `pz --version`; then add the superseded notice (rule 2) to
   the previous release when this one fixes something users of it should know about.
9. **Deprecations** follow the changelog guidance: announce in release N, remove in release N+2
   at the earliest (e.g. the `adaptive_ai_engine` alias), and say so in both releases' notes.

## Hardening note

Third-party Actions are avoided (the release is cut with the preinstalled `gh`
CLI); first-party `actions/*` are GitHub-owned. For maximum supply-chain
hardening, pin every `uses:` to a 40-char commit SHA:
`gh api repos/<owner>/<action>/commits/<tag> -q .sha`.
