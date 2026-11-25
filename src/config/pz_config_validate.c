/* Validation of one setting's value against the schema (type, range, choices). */

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config/pz_config.h"

/* Length and newline rules, common to every type. */
static int validate_text(const PzCfgKey *k, const char *value, char *err, size_t errn) {
    if (strlen(value) >= PZ_CFG_MAX_VALUE) {
        snprintf(err, errn, "%s: value too long", k->key);
        return -1;
    }
    if (strchr(value, '\n') && k->type != PZ_CFG_STRING && k->type != PZ_CFG_LIST) {
        snprintf(err, errn, "%s: value may not contain a newline", k->key);
        return -1;
    }
    return 0;
}

/* strtoll / strtod consumed the whole text and produced a finite, in-range number. */
static int number_parsed(const char *value, const char *end, double d) {
    if (end == value || *end != '\0') return 0;
    return errno != ERANGE && isfinite(d);
}

/* An integer or a number, inside the key's range when it has one. */
static int validate_number(const PzCfgKey *k, const char *value, char *err, size_t errn) {
    char *end = NULL;
    errno     = 0;
    double d  = k->type == PZ_CFG_INT ? (double)strtoll(value, &end, 10) : strtod(value, &end);
    if (!number_parsed(value, end, d)) {
        snprintf(err, errn, "%s: expected %s, got \"%s\"", k->key,
                 k->type == PZ_CFG_INT ? "an integer" : "a number", value);
        return -1;
    }
    if (k->min < k->max && (d < k->min || d > k->max)) {
        snprintf(err, errn, "%s: %s is outside %g..%g", k->key, value, k->min, k->max);
        return -1;
    }
    return 0;
}

static int validate_bool(const PzCfgKey *k, const char *value, char *err, size_t errn) {
    if (strcmp(value, "true") == 0 || strcmp(value, "false") == 0) return 0;
    snprintf(err, errn, "%s: expected true or false", k->key);
    return -1;
}

/* The value must be one of the key's "|"-separated choices. */
static int validate_choice(const PzCfgKey *k, const char *value, char *err, size_t errn) {
    size_t vl = strlen(value);
    for (const char *p = k->choices; *p;) {
        const char *bar = strchr(p, '|');
        size_t l        = bar ? (size_t)(bar - p) : strlen(p);
        if (l == vl && strncmp(p, value, l) == 0) return 0;
        if (!bar) break;
        p = bar + 1;
    }
    snprintf(err, errn, "%s: \"%s\" is not one of %s", k->key, value, k->choices);
    return -1;
}

int pz_config_validate(const PzCfgKey *k, const char *value, char *err, size_t errn) {
    if (!k || !value) {
        snprintf(err, errn, "missing value");
        return -1;
    }
    if (validate_text(k, value, err, errn) != 0) return -1;
    if ((k->type == PZ_CFG_INT || k->type == PZ_CFG_FLOAT) &&
        validate_number(k, value, err, errn) != 0)
        return -1;
    if (k->type == PZ_CFG_BOOL && validate_bool(k, value, err, errn) != 0) return -1;
    return k->choices ? validate_choice(k, value, err, errn) : 0;
}
