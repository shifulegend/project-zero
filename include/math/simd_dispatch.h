#ifndef TN_SIMD_DISPATCH_H
#define TN_SIMD_DISPATCH_H

#include "core/platform.h"
#include "math/simd_core_ops.h"
#include "math/simd_preq_ops.h"
#include "math/simd_vec_ops.h"

/**
 * SIMD runtime dispatch layer.
 *
 * Provides function pointers that are initialized at startup to the best
 * available SIMD implementation (AVX2 > scalar fallback).
 *
 * Call tn_simd_init() once at program startup before using any math ops.
 */

/**
 * Initialize SIMD dispatch. Probes CPU features and sets function pointers
 * to the best available implementation. Must be called before any math ops.
 *
 * Returns a human-readable string describing the selected backend
 * (e.g., "AVX2", "Scalar").
 *
 * TODO(QA-ISS-005): SIMD availability is currently determined at compile time
 * via preprocessor flags (#ifdef __AVX2__, etc.). On machines without the
 * required ISA but compiled with those flags, this will crash with SIGILL.
 * A future version should add runtime CPUID detection to select the correct
 * kernel at startup, independent of compile-time flags.
 */
const char *tn_simd_init(void);

/* Name of the backend selected by the last tn_simd_init() ("Scalar" before). */
const char *tn_simd_backend_name(void);

/* Activation numerics of the ternary matmul, independent of which SIMD tier runs
 * (2026-11-18). INT8 is the W1.58A8 definition (per-row abs_max int8, include/math/
 * quantize_i8.h), used by bitnet.cpp and the reference model and fastest where VNNI or
 * dotprod exist; FLOAT keeps activations in float32 (slightly more precise, slower).
 * AUTO takes INT8 where the CPU has an int8 dot product and FLOAT elsewhere, i.e. what the
 * dispatcher always did. Setting a mode re-binds the kernels (tn_simd_init). */
typedef enum { TN_ACT_AUTO = 0, TN_ACT_INT8, TN_ACT_FLOAT } TnActQuant;
int tn_act_quant_parse(const char *name, TnActQuant *out); /* 0 = ok, -1 = unknown */
const char *tn_act_quant_name(TnActQuant m);
void tn_act_quant_set(TnActQuant m);
TnActQuant tn_act_quant_get(void);
/* "int8" or "float": what the ternary matmul actually runs after tier selection. */
const char *tn_act_quant_effective(void);

#endif /* TN_SIMD_DISPATCH_H */
