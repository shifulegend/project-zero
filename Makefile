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
# -MMD -MP: emit build/**.d header-dependency files (included at the bottom)
# so editing a header rebuilds every object that includes it. 2026-10-18:
# without this, adding a field to a struct embedded elsewhere (GenerationParams
# inside ServerConfig) left stale objects with the old layout linked into the
# binary/tests — caught as an ASan SEGV reading a shifted field offset.
DEPFLAGS        = -MMD -MP
CFLAGS_COMMON   = -std=c99 -Wall -Wextra -Wpedantic -Iinclude -D_POSIX_C_SOURCE=200809L $(DARWIN_DEFS) $(DEPFLAGS)
CXXFLAGS_COMMON = -std=c++17 -Wall -Wextra -Iinclude -D_POSIX_C_SOURCE=200809L $(DARWIN_DEFS) $(DEPFLAGS)
# Version string baked into the binary (overridable; derived from git by default).
# Used by --version and the startup banner; a #ifndef fallback in main.c covers
# builds where it is not passed (e.g. plain `make debug`).
# Passed unquoted; main.c stringifies it (PZ_STRINGIFY). This avoids embedding
# quotes that get mangled through the recursive-make CFLAGS="..." layer.
PZ_VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
PZ_VERSION_DEF = -DPZ_VERSION=$(PZ_VERSION)
# Phase 38: forks/packagers point update notifications at their own releases
# (make PZ_UPDATE_REPO=owner/name); default is the upstream repo (update_check.c).
ifdef PZ_UPDATE_REPO
PZ_VERSION_DEF += -DPZ_UPDATE_REPO=$(PZ_UPDATE_REPO)
endif

CFLAGS_RELEASE   = $(CFLAGS_COMMON)   -O3 -march=native -DNDEBUG $(PZ_VERSION_DEF)
CFLAGS_DEBUG     = $(CFLAGS_COMMON)   -g -O0 -march=native -fsanitize=address -fsanitize=undefined
CXXFLAGS_RELEASE = $(CXXFLAGS_COMMON) -O3 -march=native -DNDEBUG $(PZ_VERSION_DEF)
CXXFLAGS_DEBUG   = $(CXXFLAGS_COMMON) -g -O0 -march=native -fsanitize=address -fsanitize=undefined
# TSan is incompatible with ASan/UBSan in the same binary, so it gets its own
# variant (-O1: TSan docs recommend >-O0 for usable performance under load).
CFLAGS_TSAN      = $(CFLAGS_COMMON)   -g -O1 -march=native -fsanitize=thread
CXXFLAGS_TSAN    = $(CXXFLAGS_COMMON) -g -O1 -march=native -fsanitize=thread
LDFLAGS_TSAN     = -pthread -lm -fsanitize=thread
# Coverage: instruments $(LIB_OBJS) only (via `test:`'s own default CFLAGS,
# same trick as TSan) -- the individual tests/*.c driver files still compile
# under CFLAGS_DEBUG (ASan/UBSan; gcov instrumentation and ASan/UBSan are not
# mutually exclusive, unlike TSan), so what gets measured is library code
# coverage exercised by the real ASan/UBSan-checked test binaries.
# -fprofile-update=atomic: this project's thread pool exercises the same
# source lines from multiple threads concurrently; gcov's default
# (non-atomic) counter updates race and corrupt into negative counts under
# real concurrent execution (confirmed: lcov "Unexpected negative count"
# on thread_pool.c without this flag).
CFLAGS_COVERAGE   = $(CFLAGS_COMMON)   -g -O0 -march=native --coverage -fprofile-update=atomic
CXXFLAGS_COVERAGE = $(CXXFLAGS_COMMON) -g -O0 -march=native --coverage -fprofile-update=atomic
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

TARGET = pz
# Phase 39 (2026-10-21): the binary was renamed from adaptive_ai_engine. For one
# release a symlink with the old name is kept (the program prints a deprecation
# notice when run through it); drop LEGACY_TARGET the release after.
LEGACY_TARGET = adaptive_ai_engine

.PHONY: all clean debug release dist test run-tests test-tsan coverage objs demo webui-bundle screenshots

all: $(TARGET)

# build/%.o is a shared path across release/debug/dist -- Make only tracks
# file mtimes, not which CFLAGS produced an object, so switching variants
# (e.g. `make release && make debug`) without this check silently links
# stale objects built under the PREVIOUS variant's flags: a "debug" binary
# built right after "release" would carry zero ASan/UBSan instrumentation
# despite exit 0 (confirmed 2026-09-18 via `nm | grep asan` on the result).
# Recorded in build/.variant; a mismatch forces a clean before rebuilding.
#
# 2026-10-21: the stamp also records the compiler and the host's -march=native
# ISA set. Objects are compiled with host-detected ISA flags (_HAS_*VNNI,
# -march=native), so a VM that migrates to a different CPU between builds (seen
# after a container restart: test_simd_vnni failed to link against a stale
# ternary_matmul_packed_avx_vnni.o) or a gcc->clang switch on the same variant
# silently reused objects built for the other configuration.
BUILD_STAMP := build/.variant
HOST_ISA := $(shell $(CC) -march=native -dM -E - </dev/null 2>/dev/null | \
	grep -E '__(AVX|SSE|FMA|F16C|BMI|AMX|ARM_FEATURE)' | sort | cksum | cut -d' ' -f1)
VARIANT_ID = $(1) cc=$(notdir $(CC)) isa=$(HOST_ISA) vnni=$(_HAS_AVX512VNNI)$(_HAS_AVXVNNI)
define ensure_variant
	@mkdir -p build
	@if [ -f $(BUILD_STAMP) ] && [ "$$(cat $(BUILD_STAMP))" != "$(call VARIANT_ID,$(1))" ]; then \
		echo "Build configuration changed ($$(cat $(BUILD_STAMP)) -> $(call VARIANT_ID,$(1))); removing stale objects..."; \
		rm -rf build $(TARGET) $(LEGACY_TARGET); \
		mkdir -p build; \
	fi
	@echo "$(call VARIANT_ID,$(1))" > $(BUILD_STAMP)
endef

release:
	$(call ensure_variant,release)
	$(MAKE) CFLAGS="$(CFLAGS_RELEASE)" CXXFLAGS="$(CXXFLAGS_RELEASE)" all

debug:
	$(call ensure_variant,debug)
	$(MAKE) CFLAGS="$(CFLAGS_DEBUG)" CXXFLAGS="$(CXXFLAGS_DEBUG)" LDFLAGS="$(LDFLAGS) -fsanitize=address -fsanitize=undefined" all

# Portable, statically-libstdc++/libgcc-linked x86-64 binary for distribution.
# Forces the VNNI per-file flags ON (the build host may lack VNNI) and compiles
# simd_dispatch.c with all branches present; runtime CPUID dispatch keeps it safe
# on CPUs that lack a given tier. See the per-file ISA rules below.
dist:
	$(call ensure_variant,dist)
	$(MAKE) CFLAGS="$(CFLAGS_DIST)" CXXFLAGS="$(CXXFLAGS_DIST)" LDFLAGS="$(LDFLAGS_DIST)" \
	        _HAS_AVXVNNI=1 _HAS_AVX512VNNI=1 DISPATCH_DEFS=-DTN_FORCE_DISPATCH_ALL all

# Build all object files without linking (useful when main.c doesn't exist yet)
objs: $(OBJS)

# Skip building engine executable if main.c is missing
$(TARGET): $(OBJS)
	@if [ -f src/cli/main.c ]; then \
		$(CXX) -o $@ $^ $(LDFLAGS) && ln -sf $(TARGET) $(LEGACY_TARGET); \
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

# -mf16c: the optimized row dot uses _mm_cvtph_ps for the block scale; F16C
# exists on every AVX-512VNNI CPU but is not implied by -mavx512vnni at the
# x86-64-v2 dist baseline, so it must be requested explicitly (2026-07-17).
build/math/matmul_q2_0_vnni.o: src/math/matmul_q2_0_vnni.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(if $(filter 1,$(_HAS_AVX512VNNI)),-mavx512vnni -mf16c) -c -o $@ $<

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
            math/matmul_q4k math/matmul_q4k_x8 math/matmul_q2k math/matmul_q8_0 \
            math/matmul_q5_0 math/matmul_q5_1 math/matmul_q5k math/matmul_q2_0 \
            math/matmul_f16 math/quantize_i8 math/parallel_matmul math/matmul_q6k \
            math/q8_0_act math/kvq4_avx2
AVX2_OBJS := $(addprefix build/,$(addsuffix .o,$(AVX2_TUS)))

AVX512_TUS := math/ternary_matmul_packed_avx512 math/ternary_matmul_lut_avx512bw \
              math/elementwise_avx512 \
              math/rmsnorm_avx512 math/softmax_avx512 math/kvq4_avx512
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

# Calibration decides which backends to benchmark from the same macros.
build/core/calibration.o: src/core/calibration.c
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

# When the library objects are a sanitizer build (last variant was `debug`),
# links that don't use CFLAGS_DEBUG still need the ASan/UBSan runtime.
# 2026-10-18: without this, `make debug && make test` failed to link
# test_api_server/test_q2k_matvec (undefined __asan_*/__ubsan_*); CI never hit
# it only because it runs release -> test -> debug.
SAN_LINK_FLAGS = $(if $(filter debug,$(shell cat $(BUILD_STAMP) 2>/dev/null)),-fsanitize=address -fsanitize=undefined)

# test_api_server links C++ objects (chat_template.o) so must use g++ as linker.
# Compile the C test source with release flags to avoid sanitizer symbol mismatches.
build/tests/test_api_server: tests/test_api_server.c $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_RELEASE) -c -o build/tests/test_api_server.o tests/test_api_server.c
	$(CXX) -o $@ build/tests/test_api_server.o $(LIB_OBJS) $(LDFLAGS) $(SAN_LINK_FLAGS)

# test_q2k_matvec: pure C test but links C++ chat_template.o; use CXX linker.
build/tests/test_q2k_matvec: tests/test_q2k_matvec.c $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_RELEASE) -c -o build/tests/test_q2k_matvec.o tests/test_q2k_matvec.c
	$(CXX) -o $@ build/tests/test_q2k_matvec.o $(LIB_OBJS) $(LDFLAGS) $(SAN_LINK_FLAGS)

build/tests/%: tests/%.c $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_DEBUG) -MF $@.d -o $@ $< $(LIB_OBJS) $(LDFLAGS) -lstdc++

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

# 2026-10-21: `make test` builds the library as the sanitizer (debug) variant.
# It used to link whatever objects existed -- on a fresh tree (CI) the default
# release flags -- so only the test drivers ran under ASan/UBSan, never the
# engine code they exercise. Sharing the `debug` variant means the usual
# release -> test -> debug sequence still rebuilds only once.
test:
	$(call ensure_variant,debug)
	$(MAKE) CFLAGS="$(CFLAGS_DEBUG)" CXXFLAGS="$(CXXFLAGS_DEBUG)" \
	        LDFLAGS="$(LDFLAGS) -fsanitize=address -fsanitize=undefined" run-tests

run-tests: $(LIB_OBJS) $(TEST_BINS)
	@echo "=== Running tests ==="
	@for t in $(TEST_BINS); do echo "--- $$t ---"; $$t || exit 1; done
	@echo "=== All tests passed ==="

# TS-5.3: TSan run, scoped to the highest-value concurrency surfaces (thread
# pool, parallel matmul dispatch, the API server's generation_mutex trylock
# semantics) rather than the full suite, since most tests are single-threaded
# and add nothing here. Rebuilds $(LIB_OBJS) under -fsanitize=thread (reusing
# the same per-file ISA rules as release/debug/dist via ensure_variant, since
# they already key off the ambient CFLAGS) then links each target test's
# driver file directly with CFLAGS_TSAN/LDFLAGS_TSAN -- the generic
# build/tests/% rule hardcodes CFLAGS_DEBUG (ASan/UBSan) for the driver file,
# which cannot be linked into a TSan binary.
TSAN_TESTS = audit_threadpool_stress test_threading test_moe_experts test_api_server test_q4k_x8_matmul test_q2_0_matmul test_forward test_gen_queue test_tensor_rpc test_update_check

test-tsan:
	$(call ensure_variant,tsan)
	$(MAKE) CFLAGS="$(CFLAGS_TSAN)" CXXFLAGS="$(CXXFLAGS_TSAN)" objs
	@mkdir -p build/tests
	@status=0; \
	for t in $(TSAN_TESTS); do \
		echo "=== Building $$t under TSan ==="; \
		$(CC) $(CFLAGS_TSAN) -c -o build/tests/$$t.tsan.o tests/$$t.c || exit 1; \
		$(CXX) -o build/tests/$$t.tsan build/tests/$$t.tsan.o $(LIB_OBJS) $(LDFLAGS_TSAN) || exit 1; \
		echo "=== Running $$t under TSan ==="; \
		TSAN_OPTIONS="halt_on_error=0 exitcode=1" ./build/tests/$$t.tsan || status=1; \
	done; \
	if [ $$status -eq 0 ]; then echo "=== TSan: no races reported ==="; \
	else echo "=== TSan: races reported, see output above ==="; fi; \
	exit $$status

# TS-5.5: line/branch coverage of library code, from the real test suite.
# Requires lcov (genhtml). Uses ensure_variant like release/debug/dist/tsan
# so switching from a non-instrumented build doesn't link stale objects.
# 2026-10-21: runs `run-tests`, not `test` -- `test` now switches to the debug
# variant, which would wipe the coverage objects (CI reported 40 lines covered).
# lcov 2.x fails on an exclude pattern that matched nothing (e.g. '/usr/*' when
# no system header was instrumented); lcov 1.x rejects `--ignore-errors unused`.
LCOV_IGNORE_UNUSED = $(if $(shell lcov --version 2>/dev/null | grep -E 'version [2-9]'),--ignore-errors unused)
coverage:
	$(call ensure_variant,coverage)
	$(MAKE) CFLAGS="$(CFLAGS_COVERAGE)" CXXFLAGS="$(CXXFLAGS_COVERAGE)" LDFLAGS="$(LDFLAGS) --coverage" run-tests
	@mkdir -p build/coverage
	lcov --capture --directory build --output-file build/coverage/coverage_raw.info --rc lcov_branch_coverage=1
	lcov --remove build/coverage/coverage_raw.info '/usr/*' '*/tests/*' --output-file build/coverage/coverage.info --rc lcov_branch_coverage=1 $(LCOV_IGNORE_UNUSED)
	genhtml build/coverage/coverage.info --output-directory build/coverage/html --rc lcov_branch_coverage=1
	@echo "=== Coverage summary ==="
	lcov --summary build/coverage/coverage.info --rc lcov_branch_coverage=1
	@echo "=== HTML report: build/coverage/html/index.html ==="

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

# Q2_0 VNNI kernel A/B micro-benchmark (ref vs optimized row dot in one
# binary — see docs/architecture/CEILING_CALCULATION.md §7 option A)
bench-q2: $(LIB_OBJS) tools/bench_q2_0.c
	@mkdir -p build/tools
	$(CC) $(CFLAGS_RELEASE) -o build/tools/bench_q2_0 \
	    tools/bench_q2_0.c $(LIB_OBJS) $(LDFLAGS) -lstdc++
	@build/tools/bench_q2_0

clean:
	rm -rf build $(TARGET) $(LEGACY_TARGET)

# ── K-2 R-1: LTO + PGO build pipeline ─────────────────────────────────────
#
# Usage (3-step process):
#   make pgo-generate          ← build with -fprofile-generate
#   make pgo-run               ← run 3 warmup inferences to collect .gcda files
#   make pgo-build             ← rebuild with -fprofile-use + -flto=auto
#
# After `make pgo-build`, ./$(TARGET) is the fully optimised binary.
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
	./$(TARGET) --model $(MODEL) --tokenizer $(TOKENIZER) \
	    --prompt "What is the capital of France?" \
	    --threads $(PGO_THREADS) --max-tokens $(PGO_TOKENS) > /dev/null 2>&1
	./$(TARGET) --model $(MODEL) --tokenizer $(TOKENIZER) \
	    --prompt "Explain how DNA stores genetic information." \
	    --threads $(PGO_THREADS) --max-tokens $(PGO_TOKENS) > /dev/null 2>&1
	./$(TARGET) --model $(MODEL) --tokenizer $(TOKENIZER) \
	    --prompt "Describe the process of photosynthesis." \
	    --threads $(PGO_THREADS) --max-tokens $(PGO_TOKENS) > /dev/null 2>&1
	@echo "=== Profile data written to pgo_profiles/ — run 'make pgo-build' next ==="

pgo-build:
	@echo "=== K-2 R-1: PGO Phase 3 — rebuilding with -fprofile-use + -flto ==="
	$(MAKE) clean
	$(MAKE) CFLAGS="$(CFLAGS_PGO_USE)" LDFLAGS="$(LDFLAGS_LTO)" all
	@echo "=== LTO+PGO optimised binary built: ./$(TARGET) ==="

pgo-clean:
	rm -rf pgo_profiles

# Header dependency files written by $(DEPFLAGS) (absent on a clean tree).
-include $(OBJS:.o=.d) $(TEST_BINS:=.d)
