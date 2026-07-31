/*
 * tests/test_data_url.c
 *
 * Phase 22.2 — base64 data: URL decoding for image uploads.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "api/data_url.h"

#define PASS(name)  printf("[PASS] %s\n", name)
#define FAIL(name, msg) do { printf("[FAIL] %s: %s\n", name, msg); g_failures++; } while(0)

static int g_failures = 0;

static void test_decode_no_padding(void) {
    /* "ABC" base64-encodes to "QUJD" (no padding needed, 3 bytes -> 4 chars) */
    unsigned char *data = NULL; size_t len = 0;
    TernaryError err = data_url_decode("data:image/png;base64,QUJD", &data, &len);
    if (err != TN_OK) { FAIL("decode_no_padding", "expected TN_OK"); return; }
    if (len != 3 || memcmp(data, "ABC", 3) != 0) { FAIL("decode_no_padding", "wrong decoded bytes"); free(data); return; }
    PASS("decode_no_padding");
    free(data);
}

static void test_decode_with_single_padding(void) {
    /* "AB" -> "QUI=" (one padding char) */
    unsigned char *data = NULL; size_t len = 0;
    TernaryError err = data_url_decode("data:image/jpeg;base64,QUI=", &data, &len);
    if (err != TN_OK) { FAIL("decode_with_single_padding", "expected TN_OK"); return; }
    if (len != 2 || memcmp(data, "AB", 2) != 0) { FAIL("decode_with_single_padding", "wrong decoded bytes"); free(data); return; }
    PASS("decode_with_single_padding");
    free(data);
}

static void test_decode_with_double_padding(void) {
    /* "A" -> "QQ==" (two padding chars) */
    unsigned char *data = NULL; size_t len = 0;
    TernaryError err = data_url_decode("data:image/png;base64,QQ==", &data, &len);
    if (err != TN_OK) { FAIL("decode_with_double_padding", "expected TN_OK"); return; }
    if (len != 1 || data[0] != 'A') { FAIL("decode_with_double_padding", "wrong decoded byte"); free(data); return; }
    PASS("decode_with_double_padding");
    free(data);
}

static void test_longer_payload(void) {
    /* "Hello, World!" -> "SGVsbG8sIFdvcmxkIQ==" */
    unsigned char *data = NULL; size_t len = 0;
    TernaryError err = data_url_decode("data:text/plain;base64,SGVsbG8sIFdvcmxkIQ==", &data, &len);
    if (err != TN_OK) { FAIL("longer_payload", "expected TN_OK"); return; }
    if (len != 13 || memcmp(data, "Hello, World!", 13) != 0) { FAIL("longer_payload", "wrong decoded bytes"); free(data); return; }
    PASS("longer_payload");
    free(data);
}

static void test_missing_base64_marker_rejected(void) {
    unsigned char *data = NULL; size_t len = 0;
    TernaryError err = data_url_decode("data:image/png,QUJD", &data, &len);
    if (err == TN_OK) { FAIL("missing_base64_marker_rejected", "should reject non-base64 data URL"); free(data); return; }
    PASS("missing_base64_marker_rejected");
}

static void test_not_a_data_url_rejected(void) {
    unsigned char *data = NULL; size_t len = 0;
    TernaryError err = data_url_decode("https://example.com/image.png", &data, &len);
    if (err == TN_OK) { FAIL("not_a_data_url_rejected", "should reject a plain URL"); free(data); return; }
    PASS("not_a_data_url_rejected");
}

static void test_null_args_no_crash(void) {
    unsigned char *data = NULL; size_t len = 0;
    if (data_url_decode(NULL, &data, &len) == TN_OK) { FAIL("null_args_no_crash", "NULL url should fail"); return; }
    if (data_url_decode("data:image/png;base64,QUJD", NULL, &len) == TN_OK) { FAIL("null_args_no_crash", "NULL out_data should fail"); return; }
    PASS("null_args_no_crash");
}

int main(void) {
    printf("=== Phase 22.2 Data URL Decode Tests ===\n");
    test_decode_no_padding();
    test_decode_with_single_padding();
    test_decode_with_double_padding();
    test_longer_payload();
    test_missing_base64_marker_rejected();
    test_not_a_data_url_rejected();
    test_null_args_no_crash();

    printf("\n");
    if (g_failures == 0) { printf("=== All data_url tests passed ===\n"); return 0; }
    printf("=== %d test(s) FAILED ===\n", g_failures);
    return 1;
}
