/* Strict Content-Length / Transfer-Encoding parsing. */
#include <stdio.h>
#include <string.h>
#include "api/http_headers.h"
#include "api/http_request.h"

static int g_fail;
#define CHECK(c, msg)                                                                              \
    do {                                                                                           \
        if (!(c)) {                                                                                \
            printf("FAIL: %s\n", msg);                                                             \
            g_fail++;                                                                              \
        } else                                                                                     \
            printf("ok: %s\n", msg);                                                               \
    } while (0)

static HttpFraming scan(const char *s) {
    HttpFraming f;
    http_scan_framing(s, strlen(s), &f);
    return f;
}

static void test_request_ids(void) {
    CHECK(http_request_id_valid("abc-123_x.y:z"), "a plain ID is valid");
    CHECK(!http_request_id_valid(""), "empty ID is invalid");
    CHECK(!http_request_id_valid("a b"), "a space is invalid");
    CHECK(!http_request_id_valid("a\"b") && !http_request_id_valid("a\r\nX: y"),
          "quotes and CRLF are invalid (no header or log injection)");
    char long64[65], long65[66];
    memset(long64, 'a', 64);
    long64[64] = '\0';
    memset(long65, 'a', 65);
    long65[65] = '\0';
    CHECK(http_request_id_valid(long64) && !http_request_id_valid(long65), "64 characters at most");
    char a[40], b[40];
    http_new_id(a, sizeof a);
    http_new_id(b, sizeof b);
    CHECK(strcmp(a, b) != 0 && http_request_id_valid(a), "generated IDs are unique and valid");
    HttpRequest req;
    memset(&req, 0, sizeof req);
    char hdrs[128] = "A: 1\r\n";
    http_request_attach_id(&req, hdrs, sizeof hdrs);
    CHECK(strstr(hdrs, "A: 1\r\nX-Request-ID: ") == hdrs && req.request_id[0],
          "attach appends the header and generates an ID when none was sent");
}

int main(void) {
    test_request_ids();
    size_t v = 99;
    CHECK(http_parse_content_length("123", &v) == 0 && v == 123, "plain digits");
    CHECK(http_parse_content_length("  42 \r\n", &v) == 0 && v == 42,
          "surrounding spaces and CRLF");
    CHECK(http_parse_content_length("0", &v) == 0 && v == 0, "zero");
    CHECK(http_parse_content_length("-1", &v) != 0, "negative rejected");
    CHECK(http_parse_content_length("+5", &v) != 0, "plus sign rejected");
    CHECK(http_parse_content_length("", &v) != 0, "empty rejected");
    CHECK(http_parse_content_length("12abc", &v) != 0, "trailing junk rejected");
    CHECK(http_parse_content_length("1 2", &v) != 0, "embedded space rejected");
    CHECK(http_parse_content_length("0x10", &v) != 0, "hex rejected");
    CHECK(http_parse_content_length("99999999999999999999999999", &v) != 0, "overflow rejected");

    HttpFraming f = scan("POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\n");
    CHECK(f.has_content_length && f.content_length == 5 && !f.content_length_invalid && !f.chunked,
          "single Content-Length");
    f = scan("POST / HTTP/1.1\r\ncontent-LENGTH:7\r\n");
    CHECK(f.has_content_length && f.content_length == 7, "case-insensitive name, no space");
    f = scan("POST / HTTP/1.1\r\nContent-Length: 5\r\nContent-Length: 5\r\n");
    CHECK(f.has_content_length && !f.content_length_invalid, "identical duplicates tolerated");
    f = scan("POST / HTTP/1.1\r\nContent-Length: 5\r\nContent-Length: 6\r\n");
    CHECK(f.content_length_invalid, "conflicting duplicates rejected");
    f = scan("POST / HTTP/1.1\r\nContent-Length: abc\r\n");
    CHECK(f.content_length_invalid, "non-numeric rejected");
    f = scan("POST / HTTP/1.1\r\nX-Content-Length: 99\r\nHost: x\r\n");
    CHECK(!f.has_content_length, "longer header name is not Content-Length");
    f = scan("POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n");
    CHECK(f.chunked && f.has_transfer_encoding, "chunked detected");
    f = scan("POST / HTTP/1.1\r\ntransfer-encoding: gzip, Chunked\r\n");
    CHECK(f.chunked, "chunked among several codings, any case");
    f = scan("POST / HTTP/1.1\r\nTransfer-Encoding: identity\r\n");
    CHECK(f.has_transfer_encoding && !f.chunked, "non-chunked coding is not chunked");
    f = scan("GET / HTTP/1.1\r\nHost: x\r\n");
    CHECK(!f.has_content_length && !f.chunked, "no framing headers");
    f = scan("GET /content-length: HTTP/1.1\r\nHost: x\r\n");
    CHECK(!f.has_content_length, "request line is skipped");
    printf(g_fail ? "http_headers: %d FAILED\n" : "http_headers: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
