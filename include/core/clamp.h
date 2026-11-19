#ifndef TN_CORE_CLAMP_H
#define TN_CORE_CLAMP_H

/* Clamp v into [lo, hi]. Written as the comparison chain the call sites used, so a NaN input
 * passes through unchanged (both comparisons are false). Arguments are evaluated once. */
static inline int tn_clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline float tn_clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

#endif
