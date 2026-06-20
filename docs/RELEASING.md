# Releasing project-zero — Prebuilt x86-64 Linux Binary

How a portable, prebuilt `adaptive_ai_engine` binary is built, verified, and
published as a GitHub Release. The release pipeline lives in
[`.github/workflows/release.yml`](../.github/workflows/release.yml); the build
is driven by the Makefile `dist` target.

## TL;DR — cut a release

```bash
# 1. From the default branch, at the commit you want to ship:
git tag v0.1.0           # SemVer; stay on 0.x while pre-1.0 (marked prerelease)
git push origin v0.1.0   # tag push triggers .github/workflows/release.yml
```

The workflow then: runs the full test suite (gcc + clang), builds the portable
binary (`make dist`), packages it, smoke-tests it, runs it on a clean minimal
container, attempts a real-model golden check, and publishes a **prerelease**
with the `.tar.gz` and `SHA256SUMS` attached.

To test the build **without publishing**, run the workflow via
**Actions → Release → Run workflow** (`workflow_dispatch`); the publish job only
runs on a `v*` tag push.

## What gets shipped

| Asset | Contents |
|-------|----------|
| `adaptive_ai_engine-<version>-x86_64-linux.tar.gz` | the binary + `LICENSE` + `README.md` |
| `SHA256SUMS` | checksum of the tarball (`sha256sum -c SHA256SUMS` to verify) |

## The portable build (`make dist`)

`make release` uses `-march=native` and is **not** distributable — on the
AVX-512-VNNI CI runner it would bake in AVX-512 and `SIGILL` on older CPUs. The
distribution build is therefore separate:

```bash
make dist CC=gcc      # -> ./adaptive_ai_engine (portable)
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
file ./adaptive_ai_engine                      # ELF x86-64
ldd  ./adaptive_ai_engine                      # only libc/libm/ld (no libstdc++/libgcc)
./adaptive_ai_engine --version                 # prints version + detected backend, exit 0
for b in scalar avx2 avx512f vnni; do TN_FORCE_BACKEND=$b ./adaptive_ai_engine --version; done

# Real-model golden output (network permitting):
curl -fsSL -o models/smollm2.gguf \
  https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-f16.gguf
./adaptive_ai_engine --model models/smollm2.gguf \
  --prompt "What is the capital of France?" --max-tokens 30 --temperature 0 --threads 2 | grep -i paris
```

The release workflow runs all of the above; the golden job is `continue-on-error`
because the HuggingFace download is gated by the environment's network policy.

## Versioning

- SemVer; stay on `0.MINOR.PATCH` while pre-1.0 and publish as **prerelease**.
- The tag (`vX.Y.Z`) is the version embedded in the binary (`--version`), via
  `git describe` → `-DPZ_VERSION` (Makefile / CMake).

## Hardening note

Third-party Actions are avoided (the release is cut with the preinstalled `gh`
CLI); first-party `actions/*` are GitHub-owned. For maximum supply-chain
hardening, pin every `uses:` to a 40-char commit SHA:
`gh api repos/<owner>/<action>/commits/<tag> -q .sha`.
