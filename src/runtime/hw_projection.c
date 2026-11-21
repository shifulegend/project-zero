/* hw_projection.c — the hardware profile's pre-load tok/s projections (pure functions). */
#include "runtime/hardware_profile.h"
#include "runtime/hw_model_constants.h"

/* Bytes the classifier streams per token in this format. */
double tn_hw_classifier_bytes(TnClassifierFormat fmt) {
    switch (fmt) {
        case TN_CLS_INT4:
            return (double)MODEL_VOCAB * MODEL_DIM / 2.0;
        case TN_CLS_INT8:
            return (double)MODEL_VOCAB * MODEL_DIM;
        default:
            return (double)MODEL_VOCAB * MODEL_DIM * 2.0;
    }
}

/*
 * L3-aware ceiling estimate (pre-load only; the post-load
 * set_model_bytes() correction uses plain bandwidth division).
 *
 * In steady-state autoregressive inference, L3 retains recently
 * accessed weights between tokens. The effective DRAM read per
 * token depends on how much weight data fits in L3.
 *
 * Heuristic (2026-07-17: comment corrected to match the code — it
 * previously described a "~half stay cached" model that only ever
 * existed in calibration.c's own copy):
 *  1. Classifier counted fully cached if cls_bytes < L3
 *  2. Ternary layers: min(L3 - classifier, ternary_bytes) counted
 *     cached; the remainder billed to DRAM
 *
 * Because the cache credit REDUCES billed DRAM bytes, it RAISES the
 * projected ceiling — this is an optimistic estimate of the
 * bandwidth bound, not a lower bound on performance (the previous
 * comment had that backwards; measured throughput "exceeding the
 * ceiling" was in fact a symptom of the probe under-measuring
 * bandwidth, fixed the same day — see bw_thread_fn above).
 */
double tn_hw_cached_ceiling(double l3_usable, double bw_gbps, double weight_bytes_per_tok,
                            double cls_bytes) {
    double dram_per_tok = weight_bytes_per_tok;

    if (l3_usable > 0.0 && l3_usable < dram_per_tok) {
        /* Classifier: fully cached if it fits in L3 */
        double cls_cached = 0.0;
        if (cls_bytes < l3_usable) {
            cls_cached = cls_bytes;
        }
        /* Ternary layers: cache (L3 - classifier) worth of layers */
        double l3_for_layers  = l3_usable - cls_cached;
        double ternary_cached = 0.0;
        if (l3_for_layers > 0.0 && l3_for_layers < (double)MODEL_TERNARY_BYTES) {
            ternary_cached = l3_for_layers;
        } else if (l3_for_layers >= (double)MODEL_TERNARY_BYTES) {
            ternary_cached = (double)MODEL_TERNARY_BYTES;
        }
        dram_per_tok -= (cls_cached + ternary_cached);
    } else if (l3_usable >= dram_per_tok) {
        dram_per_tok = 0.0;
    }

    if (dram_per_tok > 0.0) {
        double seconds_per_tok = dram_per_tok / (bw_gbps * 1e9);
        return 1.0 / seconds_per_tok;
    } else {
        return 999.0; /* compute-limited */
    }
}
