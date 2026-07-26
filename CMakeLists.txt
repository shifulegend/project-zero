cmake_minimum_required(VERSION 3.12)
project(adaptive_ai_engine C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Architecture auto-detection
# Prevents ARM builds from receiving x86 SIMD flags (and vice versa).
# ---------------------------------------------------------------------------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|i[3-6]86)")
    set(DEFAULT_AVX2   ON)
    set(DEFAULT_AVX512 OFF)
    set(DEFAULT_NEON   OFF)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(arm|aarch64|ARM64)")
    set(DEFAULT_AVX2   OFF)
    set(DEFAULT_AVX512 OFF)
    set(DEFAULT_NEON   ON)
else()
    set(DEFAULT_AVX2   OFF)
    set(DEFAULT_AVX512 OFF)
    set(DEFAULT_NEON   OFF)
endif()

option(ENABLE_AVX2   "Enable AVX2 SIMD kernels"    ${DEFAULT_AVX2})
option(ENABLE_AVX512 "Enable AVX-512 SIMD kernels"  ${DEFAULT_AVX512})
# Portable distribution build — mirrors the Makefile `dist` target: x86-64-v2
# baseline with per-file SIMD ISA so AVX2/AVX-512/VNNI light up at RUNTIME via
# simd_dispatch, plus static libstdc++/libgcc. The Makefile remains canonical
# for releases; this keeps CMake in sync (config rule). OFF by default so the
# security-audit CI (plain cmake) is unaffected.
option(PZ_DIST "Portable x86-64 distribution build (static libstdc++, runtime SIMD dispatch)" OFF)

# Version string baked into the binary (parity with the Makefile). main.c
# stringifies it; falls back to "dev" when git is unavailable.
execute_process(COMMAND git describe --tags --always --dirty
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                OUTPUT_VARIABLE PZ_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT PZ_VERSION)
    set(PZ_VERSION "dev")
endif()
add_compile_definitions(PZ_VERSION=${PZ_VERSION})

# In the portable build the SIMD kernels carry their own ISA per-file, so keep
# the GLOBAL flags at the baseline (no project-wide -mavx2/-mavx512).
if(PZ_DIST)
    set(ENABLE_AVX2 OFF)
    set(ENABLE_AVX512 OFF)
endif()
option(ENABLE_NEON   "Enable ARM NEON SIMD kernels"  ${DEFAULT_NEON})
option(ENABLE_TESTS  "Build unit tests"              ON)

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
if(WIN32)
    add_definitions(-DTN_WIN32=1 -DTN_POSIX=0)
else()
    add_definitions(-DTN_WIN32=0 -DTN_POSIX=1 -D_GNU_SOURCE)
endif()

# ---------------------------------------------------------------------------
# Include directories
# ---------------------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/include)

# ---------------------------------------------------------------------------
# Threading (portable — works on Linux, macOS, Alpine, MinGW, MSVC)
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
if(MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W4")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /O2 /DNDEBUG")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /Od /Zi")
    if(ENABLE_AVX2)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX2")
    elseif(ENABLE_AVX512)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX512")
    endif()
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic")
    # Elara Audit: Added OS-level stack protection, PIE, and Fortify Source
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fstack-protector-strong -fPIE")
    # -z,relro,-z,now are Linux ld-only; macOS ld ignores / rejects them
    if(NOT APPLE)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pie -Wl,-z,relro,-z,now")
    endif()

    set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -D_FORTIFY_SOURCE=2")
    set(CMAKE_C_FLAGS_DEBUG "-g -O0 -fsanitize=address -fsanitize=undefined")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-fsanitize=address -fsanitize=undefined")
    if(ENABLE_AVX2)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2 -mfma -mf16c")
    endif()
    if(ENABLE_AVX512)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx512f -mavx512bw")
    endif()
    if(ENABLE_NEON)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfpu=neon")
    endif()
    if(PZ_DIST)
        set(CMAKE_C_FLAGS_RELEASE "-O3 -march=x86-64-v2 -mtune=generic -DNDEBUG -D_FORTIFY_SOURCE=2")
    endif()
endif()

# ---------------------------------------------------------------------------
# Source files — explicit listing
#
# REGISTER: Every time a new .c file is created under src/, it MUST be added
# to the appropriate list below. Failing to do so will cause link errors.
#
# Current count: 55 source files (last updated: Phase 21)
# ---------------------------------------------------------------------------
set(CORE_SOURCES
    src/core/config.c
    src/core/debug.c
    src/core/error.c
    src/core/gguf_quant.c
    src/core/hardware_profile.c
    src/core/calibration.c
    src/core/mla_run_state.c
    src/core/moe_config.c
    src/core/moe_weights.c
    src/core/qwen35_run_state.c
    src/core/run_state.c
    src/core/step_timing.c
    src/core/weights.c
    src/core/unpack.c
    src/core/unpack_avx2.c
    src/core/gguf_reader.c
    src/core/gguf_loader.c
)

set(MATH_SOURCES
    src/math/cpu_features.c
    src/math/elementwise.c
    src/math/elementwise_avx2.c
    src/math/elementwise_avx512.c
    src/math/matmul_f16.c
    src/math/matmul_q4k.c
    src/math/matmul_q2k.c
    src/math/matmul_q5_0.c
    src/math/matmul_q5_1.c
    src/math/matmul_q8_0.c
    src/math/matmul_q5k.c
    src/math/matmul_q2_0.c
    src/math/matmul_q2_0_vnni.c
    src/math/parallel_matmul.c
    src/math/quantize_i8.c
    src/math/rmsnorm.c
    src/math/rmsnorm_avx2.c
    src/math/rmsnorm_avx512.c
    src/math/rope.c
    src/math/simd_dispatch.c
    src/math/softmax.c
    src/math/softmax_avx2.c
    src/math/softmax_avx512.c
    src/math/ternary_matmul_avx2.c
    src/math/ternary_matmul_scalar.c
    src/math/ternary_matmul_packed.c
    src/math/ternary_matmul_packed_avx2.c
    src/math/ternary_matmul_packed_avx512.c
    src/math/ternary_matmul_packed_vnni.c
    src/math/ternary_matmul_packed_avx_vnni.c
    src/math/ternary_matmul_packed_dotprod.c
    src/math/ternary_matmul_packed_vnni256.c
)

set(MULTIMODAL_SOURCES
    src/multimodal/image_load.c
    src/multimodal/patch_extract.c
    src/multimodal/vision_bridge.c
    src/multimodal/vision_encoder.c
    src/multimodal/vision_projector.c
    src/multimodal/vision_weights_load.c
    src/multimodal/vision_pipeline.c
)

set(MEMORY_SOURCES
    src/memory/aligned_alloc.c
    src/memory/mapped_file.c
)

set(THREADING_SOURCES
    src/threading/cpu_probe.c
    src/threading/thread_pool.c
)

set(TOKENIZER_SOURCES
    src/tokenizer/chat_template.cpp
    src/tokenizer/tokenizer_load.c
    src/tokenizer/tokenizer_encode.c
    src/tokenizer/tokenizer_decode.c
    src/tokenizer/tokenizer_gguf.c
)

set(TRANSFORMER_SOURCES
    src/transformer/embedding.c
    src/transformer/attention.c
    src/transformer/ffn.c
    src/transformer/forward.c
    src/transformer/generate.c
    src/transformer/mla_attention.c
    src/transformer/moe_ffn.c
    src/transformer/moe_router.c
    src/transformer/moe_scheduler.c
    src/transformer/qwen35_attention.c
)

set(SAMPLING_SOURCES
    src/sampling/rng.c
    src/sampling/argmax.c
    src/sampling/temperature.c
    src/sampling/top_p.c
    src/sampling/top_k.c
)

set(KV_CACHE_SOURCES
    src/kv_cache/kv_compress.c
    src/kv_cache/sliding_window.c
    src/kv_cache/kv_strategy.c
)

set(REASONING_SOURCES
    src/reasoning/thought_filter.c
    src/reasoning/prompt_inject.c
    src/reasoning/reasoning_generate.c
)

set(CLI_SOURCES
    src/cli/args.c
    src/cli/timer.c
    src/cli/repl.c
    src/cli/color.c
    src/cli/progress.c
    src/cli/live_stats.c
    src/cli/md_render.c
    src/cli/banner.c
)

set(API_SOURCES
    src/api/chat_compile.c
    src/api/http_server.c
    src/api/json_parse.c
    src/api/sse_stream.c
    src/api/server_config.c
    src/api/cors.c
    src/api/auth.c
    src/api/metrics.c
    src/api/openapi.c
    src/api/cancel.c
    src/api/static_assets.c
    src/api/webui_bundle_generated.c
    src/api/data_url.c
)

set(RAG_SOURCES
    src/rag/auto_retrieve.c
    src/rag/auto_save.c
    src/rag/embedder.c
    src/rag/memory_search.c
    src/rag/similarity.c
    src/rag/vector_db.c
)

set(AGENT_SOURCES
    src/agent/agent_loop.c
    src/agent/cmd_exec.c
    src/agent/output_inject.c
    src/agent/tool_interceptor.c
    src/agent/user_approval.c
)

set(SOURCES
    ${CORE_SOURCES}
    ${MATH_SOURCES}
    ${MULTIMODAL_SOURCES}
    ${MEMORY_SOURCES}
    ${THREADING_SOURCES}
    ${TOKENIZER_SOURCES}
    ${TRANSFORMER_SOURCES}
    ${SAMPLING_SOURCES}
    ${KV_CACHE_SOURCES}
    ${REASONING_SOURCES}
    ${CLI_SOURCES}
    ${API_SOURCES}
    ${RAG_SOURCES}
    ${AGENT_SOURCES}
)

# ---------------------------------------------------------------------------
# Main target
# ---------------------------------------------------------------------------
add_executable(project-zero src/cli/main.c ${SOURCES})
target_link_libraries(project-zero PRIVATE Threads::Threads m)

# Portable multiversioning (mirrors the Makefile `dist` per-file ISA rules):
# each SIMD kernel TU carries its own ISA; simd_dispatch.c stays at the baseline
# but forces all runtime branches present; the binary statically links the C++
# runtime so only libc/libm are needed at run time.
if(PZ_DIST AND NOT MSVC)
    set(PZ_AVX2_SRCS
        src/math/ternary_matmul_avx2.c src/math/ternary_matmul_packed_avx2.c
        src/math/elementwise_avx2.c src/math/rmsnorm_avx2.c src/math/softmax_avx2.c
        src/core/unpack_avx2.c
        src/math/matmul_q4k.c src/math/matmul_q2k.c src/math/matmul_q8_0.c
        src/math/matmul_q5_0.c src/math/matmul_q5_1.c src/math/matmul_q5k.c
        src/math/matmul_f16.c src/math/quantize_i8.c src/math/parallel_matmul.c)
    set(PZ_AVX512_SRCS
        src/math/ternary_matmul_packed_avx512.c src/math/elementwise_avx512.c
        src/math/rmsnorm_avx512.c src/math/softmax_avx512.c)
    set_source_files_properties(${PZ_AVX2_SRCS} PROPERTIES
        COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(${PZ_AVX512_SRCS} PROPERTIES
        COMPILE_OPTIONS "-mavx2;-mfma;-mf16c;-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq")
    # -mf16c on matmul_q2_0_vnni.c: _mm_cvtph_ps in the optimized row dot
    # (mirrors the Makefile per-file rule, 2026-07-17)
    set_source_files_properties(src/math/ternary_matmul_packed_vnni.c
        src/math/ternary_matmul_packed_vnni256.c PROPERTIES
        COMPILE_OPTIONS "-mavx512vnni;-mavx512vl")
    set_source_files_properties(src/math/matmul_q2_0_vnni.c PROPERTIES
        COMPILE_OPTIONS "-mavx512vnni;-mavx512vl;-mf16c")
    set_source_files_properties(src/math/ternary_matmul_packed_avx_vnni.c PROPERTIES
        COMPILE_OPTIONS "-mavxvnni")
    set_source_files_properties(src/math/simd_dispatch.c PROPERTIES
        COMPILE_DEFINITIONS "TN_FORCE_DISPATCH_ALL")
    target_link_options(project-zero PRIVATE -static-libstdc++ -static-libgcc)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(ENABLE_TESTS)
    enable_testing()

    # Library sources = all sources minus cli/main.c (no main.c exists yet)
    set(LIB_SOURCES ${SOURCES})
    list(FILTER LIB_SOURCES EXCLUDE REGEX ".*cli/main\\.c$")

    file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "tests/*.c")
    foreach(TEST_SRC ${TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
        add_executable(${TEST_NAME} ${TEST_SRC} ${LIB_SOURCES})
        target_link_libraries(${TEST_NAME} PRIVATE Threads::Threads m)
        # Always build tests with debug flags for sanitizer coverage
        target_compile_options(${TEST_NAME} PRIVATE
            $<$<NOT:$<C_COMPILER_ID:MSVC>>:-g -O0 -fsanitize=address -fsanitize=undefined>
        )
        target_link_options(${TEST_NAME} PRIVATE
            $<$<NOT:$<C_COMPILER_ID:MSVC>>:-fsanitize=address -fsanitize=undefined>
        )
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endforeach()
endif()
