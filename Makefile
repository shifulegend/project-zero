CC      ?= gcc
CXX     ?= g++
# On macOS, _DARWIN_C_SOURCE unlocks sysconf constants (e.g. _SC_NPROCESSORS_ONLN)
# without conflicting with strict C99 mode.
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S),Darwin)
    DARWIN_DEFS = -D_DARWIN_C_SOURCE
else
    DARWIN_DEFS =
endif
CFLAGS_COMMON   = -std=c99 -Wall -Wextra -Wpedantic -Iinclude -D_POSIX_C_SOURCE=200809L $(DARWIN_DEFS)
CXXFLAGS_COMMON = -std=c++17 -Wall -Wextra -Iinclude -D_POSIX_C_SOURCE=200809L $(DARWIN_DEFS)
# Version string baked into the binary (overridable; derived from git by default).
# Used by --version and the startup banner; a #ifndef fallback in main.c covers
# builds where it is not passed (e.g. plain `make debug`).
# Passed unquoted; main.c stringifies it (PZ_STRINGIFY). This avoids embedding
# quotes that get mangled through the recursive-make CFLAGS="..." layer.
PZ_VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
PZ_VERSION_DEF = -DPZ_VERSION=$(PZ_VERSION)

CFLAGS_RELEASE   = $(CFLAGS_COMMON)   -O3 -march=native -DNDEBUG $(PZ_VERSION_DEF)
CFLAGS_DEBUG     = $(CFLAGS_COMMON)   -g -O0 -march=native -fsanitize=address -fsanitize=undefined
CXXFLAGS_RELEASE = $(CXXFLAGS_COMMON) -O3 -march=native -DNDEBUG $(PZ_VERSION_DEF)
CXXFLAGS_DEBUG   = $(CXXFLAGS_COMMON) -g -O0 -march=native -fsanitize=address -fsanitize=undefined
LDFLAGS = -pthread -lm

# ── Portable distribution build (`make dist`) ──────────────────────────────
# One binary that runs on any x86-64-v2 (~2009+) CPU yet lights up AVX2 /
# AVX-512 / VNNI at RUNTIME via simd_dispatch. The bulk compiles at the
# portable baseline below; the SIMD kernels carry their own ISA via the
# per-file rules further down, and simd_dispatch.c is compiled at the baseline
# with -DTN_FORCE_DISPATCH_ALL so every runtime branch is present without
# emitting SIMD in the always-run startup path. Override DIST_MARCH to raise
# the floor (e.g. -march=x86-64-v3 to require AVX2 everywhere).
DIST_MARCH   ?= -march=x86-64-v2 -mtune=generic
CFLAGS_DIST   = $(CFLAGS_COMMON)   -O3 $(DIST_MARCH) -DNDEBUG $(PZ_VERSION_DEF)
CXXFLAGS_DIST = $(CXXFLAGS_COMMON) -O3 $(DIST_MARCH) -DNDEBUG $(PZ_VERSION_DEF)
LDFLAGS_DIST  = $(LDFLAGS) -static-libstdc++ -static-libgcc

# Per-TU ISA flag groups (additive: harmless under -march=native, required at
# the portable dist baseline). F16C is needed by matmul_f16; FMA by the AVX2/
# AVX-512 float paths.
# ISA flags are x86-only; leave empty on arm64/aarch64 (source files are
# guarded by TN_HAS_AVX2 / TN_HAS_AVX512 so they compile to empty TUs).
ifeq ($(filter x86_64 i686,$(UNAME_M)),)
  ISA_AVX2   =
  ISA_AVX512 =
else
  ISA_AVX2   = -mavx2 -mfma -mf16c
  ISA_AVX512 = $(ISA_AVX2) -mavx512f -mavx512bw -mavx512vl -mavx512dq
endif

# Default to release
CFLAGS   ?= $(CFLAGS_RELEASE)
CXXFLAGS ?= $(CXXFLAGS_RELEASE)

# Source discovery
SRCS     := $(shell find src -name '*.c'   2>/dev/null)
CXX_SRCS := $(shell find src -name '*.cpp' 2>/dev/null)
OBJS     := $(patsubst src/%.c,   build/%.o, $(SRCS)) \
            $(patsubst src/%.cpp, build/%.o, $(CXX_SRCS))

# Library objects (all sources except cli/main.c)
LIB_OBJS := $(filter-out build/cli/main.o, $(OBJS))

# Test sources
TEST_SRCS := $(shell find tests -name '*.c' 2>/dev/null)
TEST_BINS := $(patsubst tests/%.c, build/tests/%, $(TEST_SRCS))

TARGET = adaptive_ai_engine

.PHONY: all clean debug release dist test objs demo webui-bundle screenshots

all: $(TARGET)

release:
	$(MAKE) CFLAGS="$(CFLAGS_RELEASE)" CXXFLAGS="$(CXXFLAGS_RELEASE)" all

debug:
	$(MAKE) CFLAGS="$(CFLAGS_DEBUG)" CXXFLAGS="$(CXXFLAGS_DEBUG)" LDFLAGS="$(LDFLAGS) -fsanitize=address -fsanitize=undefined" all

# Portable, statically-libstdc++/libgcc-linked x86-64 binary for distribution.
# Forces the VNNI per-file flags ON (the build host may lack VNNI) and compiles
# simd_dispatch.c with all branches present; runtime CPUID dispatch keeps it safe
# on CPUs that lack a given tier. See the per-file ISA rules below.
dist:
	$(MAKE) CFLAGS="$(CFLAGS_DIST)" CXXFLAGS="$(CXXFLAGS_DIST)" LDFLAGS="$(LDFLAGS_DIST)" \
	        _HAS_AVXVNNI=1 _HAS_AVX512VNNI=1 DISPATCH_DEFS=-DTN_FORCE_DISPATCH_ALL all

# Build all object files without linking (useful when main.c doesn't exist yet)
objs: $(OBJS)

# Skip building engine executable if main.c is missing
$(TARGET): $(OBJS)
	@if [ -f src/cli/main.c ]; then \
		$(CXX) -o $@ $^ $(LDFLAGS); \
	else \
		echo "Skipping $(TARGET) build (src/cli/main.c not found)"; \
	fi

# ── C source pattern rule ──────────────────────────────────────────────────
build/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c -o $@ $<

# ── C++ source pattern rule ────────────────────────────────────────────────
build/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# ── Per-file SIMD flag overrides (Phase 16-S) ──────────────────────────────
#
# The AVX-VNNI kernel requires -mavxvnni which is NOT implied by -mavx2 alone.
# On Alder Lake / Raptor Lake / Zen3, -march=native includes it automatically.
# For cross-compiled portable builds that use a fixed -march (e.g. -march=haswell),
# we add it explicitly here so the kernel compiles on any GCC ≥ 12.
#
# The ARM dotprod kernel requires -march=armv8.2-a+dotprod (or -march=native
# on any ARMv8.2-A+ target).  It is guarded by #if TN_HAS_ARM_DOTPROD so it
# compiles to an empty object on x86 without error.
#
# -mavx512vnni and -mavxvnni are additive — they do not change any other
# code generation for other files in the project.

# moe_ffn.c calls madvise()/MADV_WILLNEED which require _GNU_SOURCE on Linux
# (same reason mapped_file.c has this override).
build/transformer/moe_ffn.o: src/transformer/moe_ffn.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -D_GNU_SOURCE -c -o $@ $<


# _GNU_SOURCE is a superset of _POSIX_C_SOURCE so nothing is lost.
build/memory/mapped_file.o: src/memory/mapped_file.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -D_GNU_SOURCE -c -o $@ $<

# http_server.c needs _GNU_SOURCE for strcasestr() / strncasecmp() on Linux.
build/api/http_server.o: src/api/http_server.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -D_GNU_SOURCE -c -o $@ $<

build/math/ternary_matmul_packed_avx_vnni.o: src/math/ternary_matmul_packed_avx_vnni.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(if $(filter 1,$(_HAS_AVXVNNI)),-mavxvnni) -c -o $@ $<

build/math/ternary_matmul_packed_vnni.o: src/math/ternary_matmul_packed_vnni.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(if $(filter 1,$(_HAS_AVX512VNNI)),-mavx512vnni) -c -o $@ $<

# VNNI-256: EVEX-encoded 256-bit VNNI via AVX-512VNNI — no ZMM, no frequency throttle.
# The 256-bit _mm256_dpbusds_epi32 intrinsic needs avx512vl in addition to
# avx512vnni (clang enforces this strictly; release gets it via -march=native,
# but the debug build has no -march=native so it must be passed explicitly).
build/math/ternary_matmul_packed_vnni256.o: src/math/ternary_matmul_packed_vnni256.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(if $(filter 1,$(_HAS_AVX512VNNI)),-mavx512vnni -mavx512vl) -c -o $@ $<

# ── Portable multiversioning: per-TU ISA flags (x86-64 dist) ───────────────
#
# So the bulk can compile at a portable baseline (DIST_MARCH) while AVX2 /
# AVX-512 / VNNI still light up at runtime, each SIMD kernel TU carries its own
# ISA flag here. The runtime-dispatched ops (ternary matmul, rmsnorm, softmax,
# elementwise, unpack) are guarded by cpu->* checks in simd_dispatch, so they
# are only CALLED where supported. The directly-called quant kernels
# (matmul_q*/matmul_f16/quantize_i8/parallel_matmul) have AVX2-or-scalar paths;
# building them at AVX2 sets the effective floor for quantized/dense GGUF
# models to AVX2 (BitNet-ternary text-gen still degrades to scalar at the
# baseline). These flags are additive and harmless under -march=native.
AVX2_TUS := math/ternary_matmul_avx2 math/ternary_matmul_packed_avx2 \
            math/elementwise_avx2 math/rmsnorm_avx2 math/softmax_avx2 \
            core/unpack_avx2 \
            math/matmul_q4k math/matmul_q2k math/matmul_q8_0 \
            math/matmul_q5_0 math/matmul_q5_1 math/matmul_q5k \
            math/matmul_f16 math/quantize_i8 math/parallel_matmul
AVX2_OBJS := $(addprefix build/,$(addsuffix .o,$(AVX2_TUS)))

AVX512_TUS := math/ternary_matmul_packed_avx512 math/ternary_matmul_lut_avx512bw \
              math/elementwise_avx512 \
              math/rmsnorm_avx512 math/softmax_avx512
AVX512_OBJS := $(addprefix build/,$(addsuffix .o,$(AVX512_TUS)))

$(AVX2_OBJS): build/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(ISA_AVX2) -c -o $@ $<

$(AVX512_OBJS): build/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(ISA_AVX512) -c -o $@ $<

# The dispatcher stays at the baseline ISA (no SIMD codegen in the always-run
# startup path). In the dist build, DISPATCH_DEFS=-DTN_FORCE_DISPATCH_ALL makes
# every runtime branch present even though this TU is compiled at x86-64-v2;
# the kernels it references are force-compiled above with their ISA flags.
DISPATCH_DEFS ?=
build/math/simd_dispatch.o: src/math/simd_dispatch.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DISPATCH_DEFS) -c -o $@ $<

# ── CPU SIMD capability probing (shared by lib rules + test rules) ─────────
#
# Detect VNNI support at build time so we never pass -mavx512vnni or
# -mavxvnni to a CPU that doesn't have them (prevents SIGILL and Clang errors).
ifeq ($(UNAME_S),Linux)
  _HAS_AVX512VNNI := $(shell grep -qw avx512_vnni /proc/cpuinfo && echo 1 || echo 0)
  _HAS_AVXVNNI    := $(shell grep -qw avx_vnni    /proc/cpuinfo && echo 1 || echo 0)
else ifeq ($(UNAME_S),Darwin)
  _HAS_AVX512VNNI := $(shell sysctl -n hw.optional.avx512vnni 2>/dev/null || echo 0)
  _HAS_AVXVNNI    := $(shell sysctl -n hw.optional.avxvnni    2>/dev/null || echo 0)
else
  _HAS_AVX512VNNI := 0
  _HAS_AVXVNNI    := 0
endif

# ── Test binaries ──────────────────────────────────────────────────────────
SIMD_TEST_FLAGS :=
ifeq ($(_HAS_AVX512VNNI),1)
  SIMD_TEST_FLAGS += -mavx512vnni
endif
ifeq ($(_HAS_AVXVNNI),1)
  SIMD_TEST_FLAGS += -mavxvnni
endif

# This rule must appear BEFORE the generic pattern rule to take precedence.
build/tests/test_simd_vnni: tests/test_simd_vnni.c $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_DEBUG) $(SIMD_TEST_FLAGS) -o $@ $< $(LIB_OBJS) $(LDFLAGS) -lstdc++

# test_api_server links C++ objects (chat_template.o) so must use g++ as linker.
# Compile the C test source with release flags to avoid sanitizer symbol mismatches.
build/tests/test_api_server: tests/test_api_server.c $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_RELEASE) -c -o build/tests/test_api_server.o tests/test_api_server.c
	$(CXX) -o $@ build/tests/test_api_server.o $(LIB_OBJS) $(LDFLAGS)

# test_q2k_matvec: pure C test but links C++ chat_template.o; use CXX linker.
build/tests/test_q2k_matvec: tests/test_q2k_matvec.c $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_RELEASE) -c -o build/tests/test_q2k_matvec.o tests/test_q2k_matvec.c
	$(CXX) -o $@ build/tests/test_q2k_matvec.o $(LIB_OBJS) $(LDFLAGS)

build/tests/%: tests/%.c $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_DEBUG) -o $@ $< $(LIB_OBJS) $(LDFLAGS) -lstdc++

# ── Convenience targets ────────────────────────────────────────────────────

# One-command demo: build, fetch a tiny dense GGUF (SmolLM2-135M-Instruct, F16,
# ~271 MB — tokenizer is embedded in the GGUF, so no separate file is needed),
# and run a deterministic prompt. Golden output: "The capital of France is Paris."
# The model is cached under models/ (gitignored); re-runs skip the download.
DEMO_MODEL = models/smollm2.gguf
DEMO_URL   = https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-f16.gguf

demo: release
	@mkdir -p models
	@if [ ! -f $(DEMO_MODEL) ]; then \
	    echo "=== Downloading SmolLM2-135M-Instruct (F16 GGUF, ~271 MB) ==="; \
	    if command -v curl >/dev/null 2>&1; then \
	        curl -fL --retry 3 -o $(DEMO_MODEL) "$(DEMO_URL)"; \
	    elif command -v wget >/dev/null 2>&1; then \
	        wget -O $(DEMO_MODEL) "$(DEMO_URL)"; \
	    else \
	        echo "ERROR: neither curl nor wget is available to download the demo model"; \
	        exit 1; \
	    fi; \
	else \
	    echo "=== Using cached model $(DEMO_MODEL) ==="; \
	fi
	@echo "=== Running demo: \"What is the capital of France?\" ==="
	@./$(TARGET) --model $(DEMO_MODEL) \
	    --prompt "What is the capital of France?" \
	    --max-tokens 50 --temperature 0 --threads 2

# Phase 22.2 — regenerates src/api/webui_bundle_generated.c from webui/dist/.
# Explicit, opt-in only: never part of `all`/`release`/`test`/`dist`, so
# ordinary builds and CI never need Node — the generated file is committed
# to git as the "prebuilt bundle" fallback. Run this after changing webui/src.
webui-bundle:
	npm --prefix webui ci
	npm --prefix webui run build
	@mkdir -p build/tools
	$(CC) -std=c99 -Wall -Wextra -Iinclude -o build/tools/gen_webui_bundle tools/gen_webui_bundle.c
	build/tools/gen_webui_bundle webui/dist src/api/webui_bundle_generated.c

# Phase 22.4 — regenerates design-QA reference screenshots (web UI via
# Playwright directly, CLI via `script` + xterm.js, see tools/screenshots/).
# Opt-in only: requires `npm install` in tools/screenshots/cli/ once, a
# running server for the webui capture, and the `release` binary for the
# CLI capture. Never part of `all`/`release`/`test`/CI.
screenshots:
	@echo "Run manually — see tools/screenshots/README.md for the exact steps"
	@echo "(starts a real server + drives Playwright; not automatable as a single target)."

test-packed: build/tests/test_packed_weights
	@echo "=== Running Phase 10 Packed Weight Tests ==="
	@build/tests/test_packed_weights
	@echo "=== Phase 10 Tests Passed ==="

test: $(LIB_OBJS) $(TEST_BINS)
	@echo "=== Running tests ==="
	@for t in $(TEST_BINS); do echo "--- $$t ---"; $$t || exit 1; done
	@echo "=== All tests passed ==="

bench: $(LIB_OBJS) tools/bench_simd.c
	@mkdir -p build/tools
	$(CC) $(CFLAGS_RELEASE) $(SIMD_TEST_FLAGS) -o build/tools/bench_simd \
	    tools/bench_simd.c $(LIB_OBJS) $(LDFLAGS) -lstdc++
	@echo "=== Running Phase 16-S SIMD Microbenchmarks ==="
	@build/tools/bench_simd

build/tools/bench_full: $(LIB_OBJS) tools/bench_full.c
	@mkdir -p build/tools
	$(CC) $(CFLAGS_RELEASE) -mavx512vnni -mavxvnni -o $@ \
	    tools/bench_full.c $(LIB_OBJS) $(LDFLAGS)

clean:
	rm -rf build $(TARGET)

# ── K-2 R-1: LTO + PGO build pipeline ─────────────────────────────────────
#
# Usage (3-step process):
#   make pgo-generate          ← build with -fprofile-generate
#   make pgo-run               ← run 3 warmup inferences to collect .gcda files
#   make pgo-build             ← rebuild with -fprofile-use + -flto=auto
#
# After `make pgo-build`, ./adaptive_ai_engine is the fully optimised binary.
# PGO profile data is written to pgo_profiles/ directory.
# LTO enables cross-module inlining (e.g. quantize_row_to_i8 into matmul).
#
# Note: pgo-run requires models/bitnet-b1.58-2B-4T.bin to be present.

CFLAGS_PGO_GEN  = $(CFLAGS_COMMON) -O3 -march=native -DNDEBUG \
                  -fprofile-generate=pgo_profiles
CFLAGS_PGO_USE  = $(CFLAGS_COMMON) -O3 -march=native -DNDEBUG \
                  -fprofile-use=pgo_profiles -fprofile-correction \
                  -flto=auto -fno-fat-lto-objects
LDFLAGS_LTO     = $(LDFLAGS) -flto=auto

MODEL      = models/bitnet-b1.58-2B-4T.bin
TOKENIZER  = models/bitnet-b1.58-2B-4T_tokenizer_proper.bin
PGO_THREADS = 4
PGO_TOKENS  = 25

.PHONY: pgo-generate pgo-run pgo-build pgo-clean

pgo-generate:
	@echo "=== K-2 R-1: PGO Phase 1 — building with -fprofile-generate ==="
	@mkdir -p pgo_profiles
	$(MAKE) clean
	$(MAKE) CFLAGS="$(CFLAGS_PGO_GEN)" all
	@echo "=== PGO instrumented binary ready — run 'make pgo-run' next ==="

pgo-run:
	@echo "=== K-2 R-1: PGO Phase 2 — running 3 inference passes to collect profile ==="
	@if [ ! -f $(MODEL) ]; then echo "ERROR: $(MODEL) not found"; exit 1; fi
	./adaptive_ai_engine --model $(MODEL) --tokenizer $(TOKENIZER) \
	    --prompt "What is the capital of France?" \
	    --threads $(PGO_THREADS) --max-tokens $(PGO_TOKENS) > /dev/null 2>&1
	./adaptive_ai_engine --model $(MODEL) --tokenizer $(TOKENIZER) \
	    --prompt "Explain how DNA stores genetic information." \
	    --threads $(PGO_THREADS) --max-tokens $(PGO_TOKENS) > /dev/null 2>&1
	./adaptive_ai_engine --model $(MODEL) --tokenizer $(TOKENIZER) \
	    --prompt "Describe the process of photosynthesis." \
	    --threads $(PGO_THREADS) --max-tokens $(PGO_TOKENS) > /dev/null 2>&1
	@echo "=== Profile data written to pgo_profiles/ — run 'make pgo-build' next ==="

pgo-build:
	@echo "=== K-2 R-1: PGO Phase 3 — rebuilding with -fprofile-use + -flto ==="
	$(MAKE) clean
	$(MAKE) CFLAGS="$(CFLAGS_PGO_USE)" LDFLAGS="$(LDFLAGS_LTO)" all
	@echo "=== LTO+PGO optimised binary built: ./adaptive_ai_engine ==="

pgo-clean:
	rm -rf pgo_profiles
