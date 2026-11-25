/*
 * tests/test_log.c
 *
 * The logger: text output identical to the message, JSON lines that parse, escaping of control
 * characters and quotes, redaction of credentials, level filtering, untrusted-string sanitising,
 * and whole lines from concurrent threads. stderr is redirected to a pipe to read what was written.
 */

#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "core/log.h"
#include "test_harness.h"

/* Runs `fn` with stderr pointing at a pipe; returns what it wrote in `buf`. */
static void capture(void (*fn)(void *), void *arg, char *buf, size_t cap) {
    int fds[2];
    fflush(stderr);
    if (pipe(fds) != 0) {
        buf[0] = '\0';
        return;
    }
    int saved = dup(2);
    dup2(fds[1], 2);
    close(fds[1]);
    fn(arg);
    fflush(stderr);
    dup2(saved, 2);
    close(saved);
    ssize_t n = read(fds[0], buf, cap - 1);
    close(fds[0]);
    buf[n > 0 ? n : 0] = '\0';
}

static void log_hello(void *arg) {
    (void)arg;
    tn_log(TN_LOG_WARN, "spec", "[spec] hello %d\n", 7);
}

static void test_text_is_the_message(void) {
    char out[512];
    tn_log_set_format(TN_LOG_TEXT);
    tn_log_set_level(TN_LOG_INFO);
    capture(log_hello, NULL, out, sizeof out);
    TEST_ASSERT(strcmp(out, "[spec] hello 7\n") == 0, "text mode writes the message unchanged");
}

static void test_json_line(void) {
    char out[512];
    tn_log_set_format(TN_LOG_JSON);
    capture(log_hello, NULL, out, sizeof out);
    tn_log_set_format(TN_LOG_TEXT);
    TEST_ASSERT(strncmp(out, "{\"ts\":\"", 7) == 0, "json line starts with a timestamp");
    TEST_ASSERT(strstr(out, "\"level\":\"warn\"") != NULL, "json carries the level");
    TEST_ASSERT(strstr(out, "\"component\":\"spec\"") != NULL, "json carries the component");
    TEST_ASSERT(strstr(out, "\"msg\":\"[spec] hello 7\"}\n") != NULL,
                "json message loses its trailing newline and the line ends in one");
    TEST_ASSERT(strchr(out, '\n') == out + strlen(out) - 1, "exactly one line");
}

static void test_render_escapes(void) {
    char out[256];
    tn_log_render(TN_LOG_JSON, "a\"b\\c\nd\te\x01", out, sizeof out);
    TEST_ASSERT(strcmp(out, "a\\\"b\\\\c\\nd\\te\\u0001") == 0,
                "json escapes quotes, backslash, controls");
    tn_log_render(TN_LOG_TEXT, "x\ry\x1b[31m\n", out, sizeof out);
    TEST_ASSERT(strcmp(out, "x\\x0dy\\x1b[31m\n") == 0,
                "text escapes carriage returns and escape sequences, keeps the final newline");
}

static void test_redaction(void) {
    char out[256];
    tn_log_render(TN_LOG_TEXT, "Authorization: Bearer abc.def-123 trailing\nnext\n", out,
                  sizeof out);
    TEST_ASSERT(strstr(out, "abc.def") == NULL && strstr(out, "[redacted]") != NULL,
                "an Authorization header value is redacted");
    TEST_ASSERT(strstr(out, "next") != NULL, "only that header line is cut");
    tn_log_render(TN_LOG_TEXT, "got Bearer sk-12345_abc ok\n", out, sizeof out);
    TEST_ASSERT(strstr(out, "sk-12345") == NULL && strstr(out, "Bearer [redacted] ok") != NULL,
                "a bearer token is redacted");
    tn_log_render(TN_LOG_JSON, "cfg api_key=\"s3cr3t-value\" and api-key: tok_1", out, sizeof out);
    TEST_ASSERT(strstr(out, "s3cr3t") == NULL && strstr(out, "tok_1") == NULL,
                "api_key values are redacted in both spellings");
    tn_log_render(TN_LOG_TEXT, "the bearer of news; apiary\n", out, sizeof out);
    TEST_ASSERT(strcmp(out, "the bearer of news; apiary\n") == 0, "ordinary words are not touched");
}

static void test_sanitize(void) {
    char out[64];
    size_t n = tn_log_sanitize("GET /a\r\n[ERROR] forged", out, sizeof out);
    TEST_ASSERT(strchr(out, '\n') == NULL && strchr(out, '\r') == NULL,
                "an untrusted string cannot carry a line break");
    TEST_ASSERT(strstr(out, "\\x0d\\x0a[ERROR] forged") != NULL && n == strlen(out),
                "control characters become \\xNN");
    n = tn_log_sanitize("0123456789", out, 5);
    TEST_ASSERT(n == 4 && strcmp(out, "0123") == 0, "output is bounded and terminated");
}

static void log_levels(void *arg) {
    (void)arg;
    tn_log(TN_LOG_DEBUG, "t", "debug\n");
    tn_log(TN_LOG_INFO, "t", "info\n");
    tn_log(TN_LOG_ERROR, "t", "error\n");
}

static void test_level_filter(void) {
    char out[128];
    tn_log_set_format(TN_LOG_TEXT);
    tn_log_set_level(TN_LOG_WARN);
    capture(log_levels, NULL, out, sizeof out);
    TEST_ASSERT(strcmp(out, "error\n") == 0, "only warn and error pass a warn threshold");
    tn_log_set_level(TN_LOG_DEBUG);
    capture(log_levels, NULL, out, sizeof out);
    TEST_ASSERT(strcmp(out, "debug\ninfo\nerror\n") == 0, "a debug threshold passes everything");
    tn_log_set_level(TN_LOG_INFO);
}

static void test_parse(void) {
    TnLogFormat f;
    TnLogLevel l;
    TEST_ASSERT(tn_log_parse_format("json", &f) == 0 && f == TN_LOG_JSON, "parse json");
    TEST_ASSERT(tn_log_parse_format("text", &f) == 0 && f == TN_LOG_TEXT, "parse text");
    TEST_ASSERT(tn_log_parse_format("xml", &f) != 0 && tn_log_parse_format(NULL, &f) != 0,
                "unknown format rejected");
    TEST_ASSERT(tn_log_parse_level("warn", &l) == 0 && l == TN_LOG_WARN, "parse warn");
    TEST_ASSERT(tn_log_parse_level("loud", &l) != 0, "unknown level rejected");
}

#define THREADS 8
#define LINES 200
static void *spam(void *arg) {
    int id = (int)(long)arg;
    for (int i = 0; i < LINES; i++)
        tn_log(TN_LOG_INFO, "t", "thread %d line %d padding padding padding padding padding\n", id,
               i);
    return NULL;
}

static void spam_all(void *arg) {
    (void)arg;
    pthread_t t[THREADS];
    for (long i = 0; i < THREADS; i++) pthread_create(&t[i], NULL, spam, (void *)i);
    for (int i = 0; i < THREADS; i++) pthread_join(t[i], NULL);
}

/* Reader that drains the pipe while the writers run (a pipe holds only 64 KiB). */
typedef struct {
    int fd;
    char *buf;
    size_t cap, n;
} Drain;

static void *drain_pipe(void *arg) {
    Drain *d = (Drain *)arg;
    ssize_t k;
    while (d->n + 1 < d->cap && (k = read(d->fd, d->buf + d->n, d->cap - 1 - d->n)) > 0)
        d->n += (size_t)k;
    return NULL;
}

static void test_concurrent_lines_do_not_interleave(void) {
    static char big[THREADS * LINES * 80];
    tn_log_set_format(TN_LOG_TEXT);
    int fds[2];
    if (pipe(fds) != 0) return;
    fflush(stderr);
    int saved = dup(2);
    dup2(fds[1], 2);
    close(fds[1]);
    Drain d = {fds[0], big, sizeof(big), 0};
    pthread_t reader;
    pthread_create(&reader, NULL, drain_pipe, &d);
    spam_all(NULL);
    fflush(stderr);
    dup2(saved, 2); /* closes the write end: the reader sees EOF */
    close(saved);
    pthread_join(reader, NULL);
    close(fds[0]);
    big[d.n] = '\0';

    int ok = 1, lines = 0;
    for (char *l = strtok(big, "\n"); l; l = strtok(NULL, "\n")) {
        int id, ln;
        char tail[64];
        lines++;
        if (sscanf(l, "thread %d line %d %63[a-z ]", &id, &ln, tail) != 3 ||
            strcmp(tail, "padding padding padding padding padding") != 0)
            ok = 0;
    }
    TEST_ASSERT(lines == THREADS * LINES, "every line arrived");
    TEST_ASSERT(ok, "no line was torn or interleaved");
}

int main(void) {
    RUN_TEST(test_text_is_the_message);
    RUN_TEST(test_json_line);
    RUN_TEST(test_render_escapes);
    RUN_TEST(test_redaction);
    RUN_TEST(test_sanitize);
    RUN_TEST(test_level_filter);
    RUN_TEST(test_parse);
    RUN_TEST(test_concurrent_lines_do_not_interleave);
    TEST_SUMMARY();
}
