#include "api/rate_limit.h"

#include <math.h>
#include <time.h>

void rate_limit_init(RateLimiter *rl, double rate_per_s, double burst) {
    pthread_mutex_init(&rl->mu, NULL);
    rl->rate = rate_per_s;
    rl->burst = burst < 1.0 ? 1.0 : burst;
    rl->tokens = rl->burst;
    rl->last = 0.0;
}

double rate_limit_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

int rate_limit_take(RateLimiter *rl, double now_s, int *retry_after_s) {
    if (rl->rate <= 0.0) return 1;
    pthread_mutex_lock(&rl->mu);
    if (rl->last > 0.0 && now_s > rl->last) {
        rl->tokens += (now_s - rl->last) * rl->rate;
        if (rl->tokens > rl->burst) rl->tokens = rl->burst;
    }
    rl->last = now_s;
    int ok = 0;
    if (rl->tokens >= 1.0) {
        rl->tokens -= 1.0;
        ok = 1;
    } else if (retry_after_s) {
        double wait = (1.0 - rl->tokens) / rl->rate;
        int w = (int)ceil(wait);
        *retry_after_s = w < 1 ? 1 : w;
    }
    pthread_mutex_unlock(&rl->mu);
    return ok;
}
