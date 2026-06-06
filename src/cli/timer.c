#include "cli/timer.h"

// Platform-specific high-resolution timer
#ifdef _WIN32
#include <windows.h>
int64_t timer_now_us(void) {
    LARGE_INTEGER freq;
    LARGE_INTEGER counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (int64_t)((counter.QuadPart * 1000000LL) / freq.QuadPart);
}
#else
#include <time.h>
int64_t timer_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000LL + (int64_t)ts.tv_nsec / 1000LL;
}
#endif

double timer_tokens_per_sec(int64_t start_us, int64_t end_us, int token_count) {
    if (end_us <= start_us || token_count <= 0) return 0.0;
    double elapsed_sec = (double)(end_us - start_us) / 1000000.0;
    return (double)token_count / elapsed_sec;
}
