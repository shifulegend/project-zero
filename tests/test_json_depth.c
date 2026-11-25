/* A request body nested deeper than the parser's cap is rejected; normal
 * nesting is untouched. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "api/chat_request.h"

static int g_fail;
#define CHECK(c, msg)                                                                              \
    do {                                                                                           \
        if (!(c)) {                                                                                \
            printf("FAIL: %s\n", msg);                                                             \
            g_fail++;                                                                              \
        } else                                                                                     \
            printf("ok: %s\n", msg);                                                               \
    } while (0)

static TernaryError parse_with_nesting(int depth, char open, char close) {
    size_t cap = (size_t)depth * 2 + 256;
    char *b    = (char *)malloc(cap);
    size_t n   = (size_t)snprintf(
        b, cap, "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"extra\":");
    for (int i = 0; i < depth; i++) b[n++] = open;
    for (int i = 0; i < depth; i++) b[n++] = close;
    b[n++] = '}';
    b[n]   = 0;
    ChatRequest req;
    chat_request_init(&req);
    TernaryError e = chat_request_parse(b, &req);
    chat_request_free(&req);
    free(b);
    return e;
}

int main(void) {
    CHECK(parse_with_nesting(3, '[', ']') == TN_OK, "shallow array in an unknown key is accepted");
    CHECK(parse_with_nesting(100, '[', ']') == TN_OK, "100-deep array is accepted");
    CHECK(parse_with_nesting(255, '[', ']') == TN_OK,
          "255 levels inside the request object (total depth 256, the cap) is accepted");
    CHECK(parse_with_nesting(256, '[', ']') != TN_OK,
          "256 levels inside the request object (total 257) is rejected");
    CHECK(parse_with_nesting(1000000, '[', ']') != TN_OK,
          "1,000,000-deep array is rejected quickly");
    CHECK(parse_with_nesting(257, '{', '}') != TN_OK, "257-deep object nest is rejected");
    printf(g_fail ? "json_depth: %d FAILED\n" : "json_depth: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
