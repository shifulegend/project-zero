#ifndef TN_CORE_LIMITS_H
#define TN_CORE_LIMITS_H

/* Bounds shared by the command line, the saved settings (config/) and the HTTP
 * server. They live in this leaf header so config/ does not have to include the
 * server or the reasoning code just to know a maximum. */

/* --parallel / --parallel-batch (the cancel registry holds one entry per parallel request). */
#define PZ_PARALLEL_MAX            64
#define PZ_PARALLEL_BATCH_DEFAULT  128
#define PZ_PARALLEL_BATCH_MAX      4096

/* Most best-of candidates one request may ask for. */
#define BEST_OF_N_MAX 64

#endif
