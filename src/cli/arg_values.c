#include "cli/arg_values.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>

int arg_to_int(const char *s, int *out) {
    char *end = NULL;
    errno     = 0;
    long v    = strtol(s, &end, 10);
    if (end == s || *end != '\0' || errno == ERANGE || v < INT_MIN || v > INT_MAX) return 0;
    *out = (int)v;
    return 1;
}

int arg_to_double(const char *s, double *out) {
    char *end = NULL;
    errno     = 0;
    double v  = strtod(s, &end);
    if (end == s || *end != '\0' || errno == ERANGE || !isfinite(v)) return 0;
    *out = v;
    return 1;
}
