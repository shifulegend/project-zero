#ifndef TN_CORE_TIMER_H
#define TN_CORE_TIMER_H

#include <stdint.h>

#include "core/platform.h"

// Get current high-resolution time in microseconds
int64_t timer_now_us(void);

// Calculate tokens per second based on elapsed microseconds
double timer_tokens_per_sec(int64_t start_us, int64_t end_us, int token_count);

#endif // TN_CORE_TIMER_H
