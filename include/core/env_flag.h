#ifndef TN_CORE_ENV_FLAG_H
#define TN_CORE_ENV_FLAG_H

/* Environment switches read once and cached, safe to query from any thread. The hot paths that
 * honour A/B and debugging variables (TN_PROFILE, TN_STEP_TIMING, PZ_MOE_FUSED, PZ_Q4K_VNNI,
 * PZ_Q8A_VNNI) used a plain `static int` filled on first use, which two threads could fill at the
 * same time (a data race). The cache is an atomic; the first caller reads the variable, later calls
 * are one relaxed load, and the value never changes after the first query (setenv() afterwards is
 * ignored, as before).
 *
 *     static atomic_int g_cache = TN_ENV_FLAG_UNSET;
 *     if (tn_env_on_unless_zero(&g_cache, "PZ_MOE_FUSED")) ...   // on, unless the value starts with '0'
 *     if (tn_env_on_if_one(&g_cache, "TN_PROFILE")) ...          // off, unless the value starts with '1'
 */

#include <stdatomic.h>
#include <stdlib.h>

#define TN_ENV_FLAG_UNSET (-1)

static inline int tn_env_flag_cached(atomic_int *cache, const char *name, char trigger,
                                     int when_trigger, int otherwise) {
    int v = atomic_load_explicit(cache, memory_order_relaxed);
    if (v == TN_ENV_FLAG_UNSET) {
        const char *e = getenv(name);
        v             = (e && e[0] == trigger) ? when_trigger : otherwise;
        atomic_store_explicit(cache, v, memory_order_relaxed);
    }
    return v;
}

/* 1 unless $name starts with '0'. */
static inline int tn_env_on_unless_zero(atomic_int *cache, const char *name) {
    return tn_env_flag_cached(cache, name, '0', 0, 1);
}

/* 0 unless $name starts with '1'. */
static inline int tn_env_on_if_one(atomic_int *cache, const char *name) {
    return tn_env_flag_cached(cache, name, '1', 1, 0);
}

#endif /* TN_CORE_ENV_FLAG_H */
