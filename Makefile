CC      ?= gcc
CXX     ?= g++
# On macOS, _DARWIN_C_SOURCE unlocks sysconf constants (e.g. _SC_NPROCESSORS_ONLN)
# without conflicting with strict C99 mode.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    DARWIN_DEFS = -D_DARWIN_C_SOURCE
else
    DARWIN_DEFS =
endif
CFLAGS_COMMON   = -std=c99 -Wall -Wextra -Wpedantic -Iinclude -D_POSIX_C_SOURCE=200809L $(DARWIN_DEFS)
CXXFLAGS_COMMON = -std=c++17 -Wall -Wextra -Iinclude -D_POSIX_C_SOURCE=200809L $(DARWIN_DEFS)
CFLAGS_RELEASE   = $(CFLAGS_COMMON)   -O3 -march=native -DNDEBUG
CFLAGS_DEBUG     = $(CFLAGS_COMMON)   -g -O0 -fsanitize=address -fsanitize=undefined
CXXFLAGS_RELEASE = $(CXXFLAGS_COMMON) -O3 -march=native -DNDEBUG
CXXFLAGS_DEBUG   = $(CXXFLAGS_COMMON) -g -O0 -fsanitize=address -fsanitize=undefined
LDFLAGS = -pthread -lm

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

.PHONY: all clean debug release test objs

all: $(TARGET)

release:
	$(MAKE) CFLAGS="$(CFLAGS_RELEASE)" CXXFLAGS="$(CXXFLAGS_RELEASE)" all

debug:
	$(MAKE) CFLAGS="$(CFLAGS_DEBUG)" CXXFLAGS="$(CXXFLAGS_DEBUG)" all

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
build/math/ternary_matmul_packed_vnni256.o: src/math/ternary_matmul_packed_vnni256.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(if $(filter 1,$(_HAS_AVX512VNNI)),-mavx512vnni) -c -o $@ $<

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
	$(CC) $(CFLAGS_DEBUG) $(SIMD_TEST_FLAGS) -o $@ $< $(LIB_OBJS) $(LDFLAGS)

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
	$(CXX) $(CFLAGS_DEBUG) -o $@ $< $(LIB_OBJS) $(LDFLAGS)

# ── Convenience targets ────────────────────────────────────────────────────
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
	$(CC) $(CFLAGS_RELEASE) -mavx512vnni -mavxvnni -o build/tools/bench_simd \
	    tools/bench_simd.c $(LIB_OBJS) $(LDFLAGS)
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
