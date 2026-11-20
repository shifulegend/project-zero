/*
 * test_response_format.c -- chat_request_parse() accepts response_format "text" and "json_object"
 * and refuses any other type (json_schema, unknown) instead of silently ignoring it.
 */
#include <stdio.h>
#include <string.h>

#include "api/chat_request.h"

static int g_fail;

int main(void) {
    const char *head =
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"response_format\":";
    struct {
        const char *rf;
        TernaryError want;
        int json_mode;
    } cases[] = {
        {"{\"type\":\"json_object\"}", TN_OK, 1},
        {"{\"type\":\"text\"}", TN_OK, 0},
        {"{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"x\"}}", TN_ERR_INVALID_ARGS, 0},
        {"{\"type\":\"grammar\"}", TN_ERR_INVALID_ARGS, 0},
        {"{\"type\":\"\"}", TN_ERR_INVALID_ARGS, 0},
        {"{}", TN_OK, 0},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        char json[512];
        snprintf(json, sizeof(json), "%s%s}", head, cases[i].rf);
        ChatRequest req;
        chat_request_init(&req);
        TernaryError err = chat_request_parse(json, &req);
        int ok =
            err == cases[i].want && (err != TN_OK || req.output.json_mode == cases[i].json_mode);
        printf("%s: response_format %s\n", ok ? "ok" : "FAIL", cases[i].rf);
        g_fail += !ok;
        chat_request_free(&req);
    }
    printf("=== %s ===\n", g_fail ? "FAILED" : "response_format tests passed");
    return g_fail ? 1 : 0;
}
