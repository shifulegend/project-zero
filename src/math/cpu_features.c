#include "math/cpu_features.h"
#include "core/platform.h"
#include <string.h>
#include <stdio.h>

/* ── x86-64 CPUID detection ───────────────────────────────────────────────── */
#if TN_ARCH_X86
#include <cpuid.h>

/*
 * CPUID leaf/sub-leaf/register/bit positions for each feature:
 *
 * Leaf 1, ECX:
 *   [28] AVX
 *
 * Leaf 7, sub-leaf 0, EBX:
 *   [5]  AVX2
 *   [16] AVX-512F
 *   [30] AVX-512BW
 *
 * Leaf 7, sub-leaf 0, ECX:
 *   [11] AVX-512VNNI
 *   [1]  AVX-512VBMI
 *
 * Leaf 7, sub-leaf 0, EDX:
 *   [23] AVX-512FP16
 *
 * Leaf 7, sub-leaf 1, EAX:
 *   [4]  AVX-VNNI  (256-bit VNNI without AVX-512 — Alder Lake+, Zen3+)
 *   [5]  AVX-512BF16
 */
static void detect_x86(TnCpuFeatures *f) {
    unsigned int eax, ebx, ecx, edx;

    /* Leaf 7, sub-leaf 0 — extended features */
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        f->avx2        = (ebx >> 5)  & 1;
        f->avx512f     = (ebx >> 16) & 1;
        f->avx512vnni  = (ecx >> 11) & 1;
        f->avx512vbmi  = (ecx >> 1)  & 1;
        f->avx512fp16  = (edx >> 23) & 1;
    }

    /* Leaf 7, sub-leaf 1 — additional extended features */
    if (__get_cpuid_count(7, 1, &eax, &ebx, &ecx, &edx)) {
        f->avx_vnni    = (eax >> 4) & 1;
        f->avx512bf16  = (eax >> 5) & 1;
    }
}
#endif /* TN_ARCH_X86 */

/* ── ARM feature detection ────────────────────────────────────────────────── */
#if TN_ARCH_ARM

#if TN_POSIX && !TN_APPLE
#include <sys/auxv.h>
#include <asm/hwcap.h>

/* HWCAP bit positions — from <asm/hwcap.h> on Linux ARM64 */
#ifndef HWCAP_ASIMDDP
    #define HWCAP_ASIMDDP (1 << 20)  /* SDOT/UDOT dotprod */
#endif
#ifndef HWCAP_SVE
    #define HWCAP_SVE     (1 << 22)
#endif
#ifndef HWCAP2_SVE2
    #define HWCAP2_SVE2   (1 << 1)
#endif
#ifndef HWCAP2_BF16
    #define HWCAP2_BF16   (1 << 14)
#endif

static void detect_arm_linux(TnCpuFeatures *f) {
    unsigned long hwcap  = getauxval(AT_HWCAP);
    unsigned long hwcap2 = getauxval(AT_HWCAP2);

    f->neon        = true;  /* mandatory on AArch64 */
    f->arm_dotprod = (hwcap & HWCAP_ASIMDDP) != 0;
    f->arm_sve2    = (hwcap2 & HWCAP2_SVE2)  != 0;
    f->arm_bf16    = (hwcap2 & HWCAP2_BF16)  != 0;
}
#endif /* TN_POSIX && !TN_APPLE */

#if TN_APPLE
#include <sys/sysctl.h>

static bool sysctl_bool(const char *name) {
    int val = 0;
    size_t sz = sizeof(val);
    return sysctlbyname(name, &val, &sz, NULL, 0) == 0 && val != 0;
}

static void detect_arm_apple(TnCpuFeatures *f) {
    f->neon          = true;   /* baseline on all Apple Silicon */
    f->apple_silicon = true;
    f->arm_dotprod   = sysctl_bool("hw.optional.arm.FEAT_DotProd");
    f->arm_sve2      = false;  /* Apple does not expose SVE2 */
    f->arm_bf16      = sysctl_bool("hw.optional.arm.FEAT_BF16");
}
#endif /* TN_APPLE */

#endif /* TN_ARCH_ARM */

/* ── Cached result ────────────────────────────────────────────────────────── */
static TnCpuFeatures g_features;
static bool          g_detected = false;

const TnCpuFeatures *tn_cpu_features_detect(void) {
    if (g_detected) return &g_features;

    memset(&g_features, 0, sizeof(g_features));

#if TN_ARCH_X86
    detect_x86(&g_features);
#endif

#if TN_ARCH_ARM
#  if TN_APPLE
    detect_arm_apple(&g_features);
#  elif TN_POSIX
    detect_arm_linux(&g_features);
#  endif
#endif

    /* Populate best_backend name */
    const char *name = tn_cpu_best_backend_name(&g_features);
    strncpy(g_features.best_backend, name, sizeof(g_features.best_backend) - 1);
    g_features.best_backend[sizeof(g_features.best_backend) - 1] = '\0';

    g_detected = true;
    return &g_features;
}

const char *tn_cpu_best_backend_name(const TnCpuFeatures *f) {
    /* x86 priority: AVX-512 VNNI > AVX-VNNI > AVX-512F > AVX2 > Scalar */
    if (f->avx512vnni) return "AVX-512 VNNI";
    if (f->avx_vnni)   return "AVX-VNNI";
    if (f->avx512f)    return "AVX-512F";
    if (f->avx2)       return "AVX2";

    /* ARM priority: DOTPROD > NEON > Scalar */
    if (f->arm_dotprod) return "NEON+dotprod";
    if (f->neon)        return "NEON";

    return "Scalar";
}

void tn_cpu_features_report(const TnCpuFeatures *f) {
    printf("CPU Feature Detection:\n");
#if TN_ARCH_X86
    printf("  [x86-64]\n");
    printf("  AVX2         : %s\n", f->avx2        ? "YES" : "no");
    printf("  AVX-512F     : %s\n", f->avx512f     ? "YES" : "no");
    printf("  AVX-512 VNNI : %s  <-- ternary int8 kernel (64 MACs/cycle)\n",
           f->avx512vnni  ? "YES" : "no");
    printf("  AVX-VNNI     : %s  <-- 256-bit int8 kernel (32 MACs/cycle)\n",
           f->avx_vnni    ? "YES" : "no");
    printf("  AVX-512 BF16 : %s\n", f->avx512bf16  ? "YES" : "no");
    printf("  AVX-512 FP16 : %s\n", f->avx512fp16  ? "YES" : "no");
    printf("  AVX-512 VBMI : %s\n", f->avx512vbmi  ? "YES" : "no");
#endif
#if TN_ARCH_ARM
    printf("  [ARM]\n");
    printf("  NEON         : %s\n", f->neon         ? "YES" : "no");
    printf("  Dotprod      : %s  <-- SDOT int8 kernel (ARM VNNI equiv)\n",
           f->arm_dotprod  ? "YES" : "no");
    printf("  SVE2         : %s\n", f->arm_sve2     ? "YES" : "no");
    printf("  BF16         : %s\n", f->arm_bf16     ? "YES" : "no");
    printf("  Apple Silicon: %s\n", f->apple_silicon? "YES" : "no");
#endif
    printf("  Best backend : %s\n", tn_cpu_best_backend_name(f));
}
