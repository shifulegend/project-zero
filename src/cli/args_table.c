/*
 * src/cli/args_table.c
 *
 * The engine behind the table-driven flag groups (include/cli/args_table.h).
 */

#include "cli/args_table.h"

#include <string.h>

#include "cli/arg_values.h"
#include "core/log.h"

static int in_range(const ArgFlagSpec *f, double v) {
    if (f->range == ARG_R_CLOSED) return v >= f->lo && v <= f->hi;
    if (f->range == ARG_R_OPEN_LOW) return v > f->lo && v <= f->hi;
    return 1;
}

/* Parses `text` into the field as the flag's kind; *as_double is the stored value for the range
 * check (a float field is compared as the float it holds). 0 on a malformed number. */
static int store_number(const ArgFlagSpec *f, char *dst, const char *text, double *as_double) {
    if (f->kind == ARG_K_INT) {
        int v;
        if (!arg_to_int(text, &v)) return 0;
        *(int *)dst = v;
        *as_double  = v;
        return 1;
    }
    double d;
    if (!arg_to_double(text, &d)) return 0;
    if (f->kind == ARG_K_FLOAT) {
        *(float *)dst = (float)d;
        *as_double    = *(float *)dst;
    } else {
        *(double *)dst = d;
        *as_double     = d;
    }
    return 1;
}

static TernaryError apply_flag(const ArgFlagSpec *f, const char *tag, CliArgs *args,
                               const char *flag, const char *text) {
    char *dst = (char *)args + f->offset;
    if (f->kind == ARG_K_STR) {
        *(char **)dst = (char *)text;
        return TN_OK;
    }
    double v;
    if (!store_number(f, dst, text, &v)) {
        TN_LOG_ERR("args", "Error: %s expects %s, got '%s'\n", flag,
                   f->kind == ARG_K_INT ? "an integer" : "a number", text);
        return TN_ERR_INVALID_CONFIG;
    }
    if (in_range(f, v)) return TN_OK;
    TN_LOG_ERR(tag, "Error: %s\n", f->message);
    return TN_ERR_INVALID_CONFIG;
}

TernaryError args_table_parse(const ArgFlagSpec *table, size_t n, const char *tag, CliArgs *args,
                              int argc, char **argv, int *ip, int *hit) {
    int i = *ip;
    for (size_t k = 0; k < n; k++) {
        const ArgFlagSpec *f = &table[k];
        if (strcmp(argv[i], f->flag) != 0) continue;
        if (f->kind == ARG_K_SET) {
            *(bool *)((char *)args + f->offset) = true;
            *hit                                = 1;
            return TN_OK;
        }
        if (i + 1 >= argc) return TN_OK;
        TernaryError err = apply_flag(f, tag, args, argv[i], argv[i + 1]);
        if (err != TN_OK) return err;
        *ip  = i + 1;
        *hit = 1;
        return TN_OK;
    }
    return TN_OK;
}
