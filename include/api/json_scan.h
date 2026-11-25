#ifndef TN_API_JSON_SCAN_H
#define TN_API_JSON_SCAN_H

/*
 * json_scan.h — scanning primitives of the hand-rolled JSON reader. Each takes a cursor
 * (`const char **p`) that it advances. Shared by api/json_parse.c, api/chat_request_parse.c and
 * the generic top-level field access; not a general-purpose JSON library.
 */

#include <stddef.h>

#include "api/chat_request.h"
#include "core/error.h"

void json_skip_ws(const char **p);
/* Skips whitespace, then consumes `c`; 0 when the next character is something else. */
int json_expect_char(const char **p, char c);
/* Reads a quoted string into dst (NUL-terminated, truncated to dst_cap); 0 when malformed. */
int json_parse_string(const char **p, char *dst, size_t dst_cap);
/* Same, into a malloc'd string of any length; NULL when malformed. */
char *json_parse_string_heap(const char **p);
double json_parse_number(const char **p);
/* 1 / 0 for true / false, -1 when the value is not a boolean. */
int json_parse_bool(const char **p);
/* Skips one whole value of any type, nested arrays and objects included. */
void json_skip_value(const char **p);

/* "messages": [...] of the chat request (plain strings or content-part arrays); the cursor is at
 * the opening '['. */
TernaryError json_parse_messages(const char **p, ChatRequest *req);

#endif /* TN_API_JSON_SCAN_H */
