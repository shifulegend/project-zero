#ifndef TN_CLI_ARGS_TABLE_H
#define TN_CLI_ARGS_TABLE_H

/* Table-driven command-line flags: one row per flag (name, field inside CliArgs, kind, optional
 * range) instead of one else-if branch each. Used by the flag groups of parse_args() (args.c); the
 * behaviour of every row is the one its branch had, and tests/test_args_golden.c pins it. */

#include <stddef.h>

#include "cli/args.h"
#include "core/error.h"

typedef enum {
    ARG_K_STR,    /* char * field kept as given (a pointer into argv) */
    ARG_K_INT,    /* int field, strict integer */
    ARG_K_FLOAT,  /* float field, strict number */
    ARG_K_DOUBLE, /* double field, strict number */
    ARG_K_SET     /* bool field set to true; the flag takes no value */
} ArgKind;

typedef enum {
    ARG_R_NONE,    /* any value */
    ARG_R_CLOSED,  /* lo <= v <= hi */
    ARG_R_OPEN_LOW /* lo < v <= hi */
} ArgRange;

typedef struct {
    const char *flag;
    size_t offset; /* of the field inside CliArgs */
    ArgKind kind;
    ArgRange range;
    double lo, hi;
    const char *message; /* printed after "Error: " when the value is out of range */
} ArgFlagSpec;

#define TN_ARG_STR_(x) #x
#define TN_ARG_STR(x) TN_ARG_STR_(x) /* a macro's value as text, for range messages */

/* If argv[*ip] is one of the table's flags (and, for a valued one, has a value after it): applies
 * it, advances *ip past the value and sets *hit. A bad or out-of-range value returns
 * TN_ERR_INVALID_CONFIG (logged under `tag`). Nothing matched: returns TN_OK with *hit left alone.
 */
TernaryError args_table_parse(const ArgFlagSpec *table, size_t n, const char *tag, CliArgs *args,
                              int argc, char **argv, int *ip, int *hit);

/* The tail every group parser shares: an error, or no match, leaves *ip alone; a match advances it
 * to the last argument used and marks the flag handled. */
static inline TernaryError args_group_result(TernaryError err, int hit, int i, int *ip,
                                             int *handled) {
    if (err != TN_OK || !hit) return err;
    *ip      = i;
    *handled = 1;
    return TN_OK;
}

#endif /* TN_CLI_ARGS_TABLE_H */
