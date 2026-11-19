#ifndef TN_API_RATE_LIMIT_H
#define TN_API_RATE_LIMIT_H

#include <pthread.h>

/* Token bucket: `rate` tokens per second, holding at most `burst`. One bucket
 * guards the cheap non-generation routes so a client (or a web page firing
 * requests at 127.0.0.1) cannot hammer them. rate <= 0 disables the limiter. */
typedef struct {
    pthread_mutex_t mu;
    double tokens;
    double rate;
    double burst;
    double last; /* seconds, monotonic; 0 until the first take */
} RateLimiter;

void rate_limit_init(RateLimiter *rl, double rate_per_s, double burst);

/* Take one token at time `now_s` (monotonic seconds). Returns 1 when allowed.
 * Returns 0 when the bucket is empty and sets *retry_after_s (>= 1) to the
 * whole seconds to wait. `now_s` is a parameter so tests control the clock. */
int rate_limit_take(RateLimiter *rl, double now_s, int *retry_after_s);

/* Current monotonic time in seconds. */
double rate_limit_now(void);

#endif
