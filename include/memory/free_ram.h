#ifndef TN_MEMORY_FREE_RAM_H
#define TN_MEMORY_FREE_RAM_H

#include "core/platform.h"

/**
 * Probe available system RAM.
 * Returns bytes of free RAM, or -1 on failure.
 */
tn_i64 tn_get_free_ram(void);

#endif /* TN_MEMORY_FREE_RAM_H */
