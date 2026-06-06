#ifndef TN_CPU_PROBE_H
#define TN_CPU_PROBE_H

/**
 * CPU topology probing for optimal thread count.
 *
 * Returns the number of threads to use for parallel work.
 * Heuristic: max(1, online_cores - 1) to leave one core free
 * for the OS and I/O.
 */
int tn_get_optimal_thread_count(void);

#endif /* TN_CPU_PROBE_H */
