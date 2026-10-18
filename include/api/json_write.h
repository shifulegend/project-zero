#ifndef TN_JSON_WRITE_H
#define TN_JSON_WRITE_H

/*
 * json_write.h — tiny growable JSON text builder for API responses.
 * Strings are escaped per RFC 8259; any allocation failure latches `err`
 * and later calls become no-ops, so callers check once at the end.
 */

#include <stddef.h>

typedef struct {
    char  *buf;
    size_t len, cap;
    int    err;
} JsonBuf;

void  jb_init(JsonBuf *jb);
void  jb_raw(JsonBuf *jb, const char *text);          /* verbatim */
void  jb_str(JsonBuf *jb, const char *s);             /* "escaped" (NULL -> null) */
void  jb_key(JsonBuf *jb, const char *key);           /* "key": */
void  jb_int(JsonBuf *jb, long long v);
void  jb_num(JsonBuf *jb, double v);                  /* NaN/inf -> null */
void  jb_bool(JsonBuf *jb, int v);
/* Returns the heap string (caller frees) or NULL if any step failed. */
char *jb_finish(JsonBuf *jb);
void  jb_free(JsonBuf *jb);

#endif /* TN_JSON_WRITE_H */
