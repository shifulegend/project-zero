/*
 * core/escape — the one-character string escapes shared by the JSON request parser
 * (src/api/json_parse.c) and the TOML config reader (src/config/pz_config.c).
 */
#include "core/escape.h"

int tn_escape_simple(char e) {
    switch (e) {
        case '"':
            return '"';
        case '\\':
            return '\\';
        case 'n':
            return '\n';
        case 'r':
            return '\r';
        case 't':
            return '\t';
        case 'b':
            return '\b';
        case 'f':
            return '\f';
        default:
            return -1;
    }
}
