#ifndef TN_CORE_LOG_H
#define TN_CORE_LOG_H

/* One place every diagnostic goes through (stderr). Text format (the default) writes the message
 * exactly as given, so existing output is unchanged; JSON format writes one object per line:
 *   {"ts":"2026-11-20T12:00:00.123Z","level":"warn","component":"gguf_loader","msg":"..."}
 * Both formats redact credentials (an "Authorization:" header, "Bearer <token>",
 * "api_key=<value>") and escape control characters. JSON escapes every one, so a value cannot
 * forge a second log line; text keeps the newlines and tabs a call site wrote (multi-line
 * messages), so a call site that logs an untrusted string passes it through tn_log_sanitize()
 * first. Each call is one write, so lines from threads never interleave. Program output (generated text, --help, tables) is not logging and stays on stdout. */

#include <stdarg.h>
#include <stddef.h>

typedef enum { TN_LOG_DEBUG = 0, TN_LOG_INFO, TN_LOG_WARN, TN_LOG_ERROR } TnLogLevel;
typedef enum { TN_LOG_TEXT = 0, TN_LOG_JSON } TnLogFormat;

/* Output format and the lowest level written (default text, TN_LOG_INFO). Thread-safe. */
void tn_log_set_format(TnLogFormat fmt);
TnLogFormat tn_log_get_format(void);
void tn_log_set_level(TnLogLevel lvl);

/* Parses "text" / "json" and "debug" / "info" / "warn" / "error"; return 0 on success. */
int tn_log_parse_format(const char *s, TnLogFormat *out);
int tn_log_parse_level(const char *s, TnLogLevel *out);
const char *tn_log_level_name(TnLogLevel lvl);

/* printf-style entry point. `component` names the subsystem ("gguf_loader", "spec"); the message
 * keeps its own "[tag] " prefix and trailing newline in text mode. */
void tn_log(TnLogLevel lvl, const char *component, const char *fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 3, 4)))
#endif
    ;
void tn_logv(TnLogLevel lvl, const char *component, const char *fmt, va_list ap);

#define TN_LOG_ERR(comp, ...) tn_log(TN_LOG_ERROR, (comp), __VA_ARGS__)
#define TN_LOG_WRN(comp, ...) tn_log(TN_LOG_WARN, (comp), __VA_ARGS__)
#define TN_LOG_INF(comp, ...) tn_log(TN_LOG_INFO, (comp), __VA_ARGS__)
#define TN_LOG_DBG(comp, ...) tn_log(TN_LOG_DEBUG, (comp), __VA_ARGS__)

/* Copies an untrusted string (a request path, a header, a file name from the network) into `out`
 * with every control character, newlines included, replaced by \xNN; always NUL-terminated;
 * returns the length written. */
size_t tn_log_sanitize(const char *s, char *out, size_t cap);

/* Pure helpers (exposed for tests): the message with credentials redacted and control characters
 * escaped (text) or JSON-escaped (json), written into `out` (always NUL-terminated); returns the
 * length written. A newline that ends the message is kept in text mode and dropped in JSON. */
size_t tn_log_render(TnLogFormat fmt, const char *msg, char *out, size_t cap);

#endif
