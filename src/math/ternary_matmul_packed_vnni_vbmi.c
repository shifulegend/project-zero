/*
 * ternary_matmul_packed_vnni_vbmi.c — the AVX-512 VNNI ternary kernels compiled
 * with AVX-512 VBMI enabled (2026-11-18).
 *
 * ternary_matmul_packed_vnni.c holds two bit-unpack paths: a 3-instruction VBMI
 * one (about 2.7x faster) and an SSE one. The VBMI path is behind a compile-time
 * macro, and the portable `make dist` build compiles that file without VBMI, so
 * the released binary always took the SSE path: BitNet 2B decoded at 15.9 tok/s
 * against 24.6 for a native build on the same host (docs/ai/mistakes.md).
 * Enabling VBMI for the whole file is not safe (the compiler could emit VBMI
 * anywhere, and a VNNI CPU without VBMI such as Cascade Lake would take SIGILL),
 * so the kernel is compiled a second time here, with its names suffixed _vbmi,
 * and tn_simd_init() binds this pair only when cpu->avx512vbmi (execution-
 * verified, math/cpu_features.c) is set. Built in every x86 build (Makefile and
 * CMake per-file flags: -mavx512vnni -mavx512vbmi).
 */
#include "core/platform.h"

#if TN_HAS_AVX512VNNI && TN_HAS_AVX512VBMI
#define TN_VNNI_KERNEL(name) name##_vbmi
#define TN_VNNI_ASSUME_VBMI 1
#include "ternary_matmul_packed_vnni.c"
#endif
