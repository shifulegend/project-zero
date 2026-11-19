/* Token bucket with an injected clock: burst, refill, cap, retry hint, disabled. */
#include <stdio.h>
#include "api/rate_limit.h"

static int g_fail;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); g_fail++; } else printf("ok: %s\n", msg); } while (0)

int main(void) {
    RateLimiter rl; int retry = 0;
    rate_limit_init(&rl, 2.0, 4.0);                /* 2 per second, burst 4 */
    int allowed = 0;
    for (int i = 0; i < 10; i++) allowed += rate_limit_take(&rl, 100.0, &retry);
    CHECK(allowed == 4, "a burst of 10 at one instant admits exactly the burst size (4)");
    CHECK(rate_limit_take(&rl, 100.0, &retry) == 0 && retry >= 1, "empty bucket refuses and gives a Retry-After of at least 1 s");
    CHECK(rate_limit_take(&rl, 100.4, &retry) == 0, "0.4 s later (0.8 tokens) still refused");
    CHECK(rate_limit_take(&rl, 100.5, &retry) == 1, "0.5 s after emptying, one token has refilled");
    CHECK(rate_limit_take(&rl, 100.5, &retry) == 0, "and only one");
    allowed = 0;
    for (int i = 0; i < 10; i++) allowed += rate_limit_take(&rl, 1000.0, &retry);
    CHECK(allowed == 4, "a long idle period refills only up to the burst, not beyond");

    RateLimiter off; rate_limit_init(&off, 0.0, 0.0);
    int all = 1;
    for (int i = 0; i < 1000; i++) all &= rate_limit_take(&off, 5.0, &retry);
    CHECK(all, "rate 0 disables the limiter");
    CHECK(rate_limit_now() > 0.0, "monotonic clock reads");
    printf(g_fail ? "rate_limit: %d FAILED\n" : "rate_limit: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
