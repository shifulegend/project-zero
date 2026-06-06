#ifndef TN_PLATFORM_H
#define TN_PLATFORM_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* ── Platform detection ────────────────────────────────────────────────────── */
#if defined(_WIN32) || defined(_WIN64)
    #define TN_WIN32 1
    #define TN_POSIX 0
#elif defined(__APPLE__) || defined(__linux__) || defined(__unix__)
    #define TN_WIN32 0
    #define TN_POSIX 1
#else
    #error "Unsupported platform"
#endif

#if defined(__APPLE__)
    #define TN_APPLE 1
#else
    #define TN_APPLE 0
#endif

/* ── Fixed-width type aliases ──────────────────────────────────────────────── */
typedef int8_t   tn_i8;
typedef int16_t  tn_i16;
typedef int32_t  tn_i32;
typedef int64_t  tn_i64;
typedef uint8_t  tn_u8;
typedef uint16_t tn_u16;
typedef uint32_t tn_u32;
typedef uint64_t tn_u64;

/* ── Compile-time SIMD feature detection ──────────────────────────────────────
 *
 * These macros reflect what the COMPILER can generate (set by -march=native
 * or explicit -mavx512vnni etc.).  They do NOT guarantee the binary will run
 * on a different machine — use the runtime TnCpuFeatures struct from
 * cpu_features.h for deployment decisions.
 *
 * x86-64 tier hierarchy (highest → lowest throughput):
 *   AVX-512 VNNI  → Ice Lake, Tiger Lake, Zen 4, Sapphire Rapids (2019+)
 *                   64 int8 MACs/cycle per core
 *   AVX-VNNI      → Alder Lake, Raptor Lake, Zen 3 (2021+) [256-bit VNNI, NO AVX-512]
 *                   32 int8 MACs/cycle per core
 *   AVX-512F      → Skylake-X, Ice Lake, Tiger Lake (2017+)
 *                   16 float32 MACs/cycle per core
 *   AVX2          → Haswell, Ryzen (2013+)
 *                   8 float32 MACs/cycle per core
 *   SSE4.2        → Nehalem (2008+) — scalar fallback
 *
 * ARM tier hierarchy:
 *   SVE2          → Cortex-X4+, Neoverse V2, Apple M4+ (2023+)
 *   DOTPROD       → Cortex-A75+, Apple M1-M4, Snapdragon 8 Gen 1+ (2017+)
 *                   16 int8 MACs/cycle per core (SDOT)
 *   NEON          → ARMv7-A / ARMv8-A baseline (2011+)
 *                   4 float32 MACs/cycle per core
 * ────────────────────────────────────────────────────────────────────────── */

/* x86-64: AVX-512 VNNI — Ice Lake+, Tiger Lake+, Zen 4+, Sapphire Rapids+ */
#if defined(__AVX512VNNI__)
    #define TN_HAS_AVX512VNNI 1
#else
    #define TN_HAS_AVX512VNNI 0
#endif

/* x86-64: AVX-VNNI (256-bit VNNI without AVX-512) — Alder Lake 12th Gen+, Zen 3+ */
#if defined(__AVXVNNI__)
    #define TN_HAS_AVXVNNI 1
#else
    #define TN_HAS_AVXVNNI 0
#endif

/* x86-64: AVX-512 VBMI — Ice Lake+, Tiger Lake+, Sapphire Rapids+ */
#if defined(__AVX512VBMI__)
    #define TN_HAS_AVX512VBMI 1
#else
    #define TN_HAS_AVX512VBMI 0
#endif

/* x86-64: AVX-512F baseline */
#if defined(__AVX512F__)
    #define TN_HAS_AVX512 1
#else
    #define TN_HAS_AVX512 0
#endif

/* x86-64: AVX2 */
#if defined(__AVX2__)
    #define TN_HAS_AVX2 1
#else
    #define TN_HAS_AVX2 0
#endif

/* x86-64: AVX-512 BF16 — Sapphire Rapids, Zen 4+ only */
#if defined(__AVX512BF16__)
    #define TN_HAS_AVX512BF16 1
#else
    #define TN_HAS_AVX512BF16 0
#endif

/* x86-64: AVX-512 FP16 — Sapphire Rapids, Meteor Lake+ only */
#if defined(__AVX512FP16__)
    #define TN_HAS_AVX512FP16 1
#else
    #define TN_HAS_AVX512FP16 0
#endif

/* ARM: NEON baseline */
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define TN_HAS_NEON 1
#else
    #define TN_HAS_NEON 0
#endif

/* ARM: dotprod extension — vdotq_s32 / vdotq_u32 (Cortex-A75+, Apple M1+) */
#if defined(__ARM_FEATURE_DOTPROD)
    #define TN_HAS_ARM_DOTPROD 1
#else
    #define TN_HAS_ARM_DOTPROD 0
#endif

/* ARM: SVE2 — scalable vectors (Cortex-X4+, Apple M4+) */
#if defined(__ARM_FEATURE_SVE2)
    #define TN_HAS_ARM_SVE2 1
#else
    #define TN_HAS_ARM_SVE2 0
#endif

/* ── CPU architecture family ──────────────────────────────────────────────── */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define TN_ARCH_X86 1
    #define TN_ARCH_ARM 0
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__)
    #define TN_ARCH_X86 0
    #define TN_ARCH_ARM 1
#else
    #define TN_ARCH_X86 0
    #define TN_ARCH_ARM 0
#endif

/* ── Portable software prefetch ───────────────────────────────────────────── */
/* _mm_prefetch/_MM_HINT_T1 are x86-only (xmmintrin.h).  Use __builtin_prefetch
 * everywhere so the same code compiles on ARM (macOS M-series CI runners) and
 * any other target without dragging in x86 intrinsic headers outside AVX guards.
 *   __builtin_prefetch(addr, rw=0, locality=2)  ≈  _mm_prefetch(addr, _MM_HINT_T1) */
#define TN_PREFETCH_T1(addr) __builtin_prefetch((const void *)(addr), 0, 2)

/* ── Binary file format constants ─────────────────────────────────────────── */
#define TN_MAGIC       0x594E5254  /* "TNRY" in little-endian */
#define TN_VERSION     1
#define TN_VDB_MAGIC   0x42445256  /* "VRDB" in little-endian */
#define TN_VDB_VERSION 1

/* ── SIMD alignment — 64 bytes covers AVX-512 cache lines ─────────────────── */
#define TN_SIMD_ALIGN 64

#endif /* TN_PLATFORM_H */
