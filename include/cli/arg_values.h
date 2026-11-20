#ifndef TN_CLI_ARG_VALUES_H
#define TN_CLI_ARG_VALUES_H

/* Strict numeric flag values: "--rate-limit abc" is an error, not 0 (which would silently turn the
 * limiter off); trailing junk and out-of-range numbers are refused too (2026-11-19). */

#include <stdio.h>

#include "core/log.h"

/* 1 and *out set when the whole string is a valid number in range; 0 otherwise. */
int arg_to_int(const char *s, int *out);
int arg_to_double(const char *s, double *out);

/* Parses argv[i + 1] into dst and advances i; returns TN_ERR_INVALID_CONFIG from the caller on a
 * bad value. Used inside parse_args(), where `i`, `argv` and `args` are in scope. */
#define ARG_INT(dst)                                                                               \
    do {                                                                                           \
        int v_;                                                                                    \
        if (!arg_to_int(argv[i + 1], &v_)) {                                                       \
            TN_LOG_ERR("args", "Error: %s expects an integer, got '%s'\n", argv[i], argv[i + 1]);  \
            return TN_ERR_INVALID_CONFIG;                                                          \
        }                                                                                          \
        (dst) = v_;                                                                                \
        i++;                                                                                       \
    } while (0)

#define ARG_NUM(dst)                                                                               \
    do {                                                                                           \
        double v_;                                                                                 \
        if (!arg_to_double(argv[i + 1], &v_)) {                                                    \
            TN_LOG_ERR("args", "Error: %s expects a number, got '%s'\n", argv[i], argv[i + 1]);    \
            return TN_ERR_INVALID_CONFIG;                                                          \
        }                                                                                          \
        (dst) = (__typeof__(dst))v_;                                                               \
        i++;                                                                                       \
    } while (0)


#endif
