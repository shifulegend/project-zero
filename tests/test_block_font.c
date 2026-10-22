/*
 * tests/test_block_font.c — the shared block wordmark font (brand/block_font.c)
 * used by the CLI banner and the web UI logo: golden "PROJECT ZERO" rendering
 * (the brand's reference ASCII), case/separator handling, raw mode, refusal of
 * unsupported characters and too-small buffers, and parity with the banner.
 */
#include "test_harness.h"
#include "brand/block_font.h"
#include "cli/banner.h"
#include <string.h>

static const char *GOLDEN =
    "####  ####   ###    ### #####  #### #####     ##### ##### ####   ### \n"
    "#  #  #  #  #   #    #  #     #       #          #  #     #  #  #   #\n"
    "####  ####  #   #    #  ###   #       #         #   ###   ####  #   #\n"
    "#     # #   #   # #  #  #     #       #        #    #     # #   #   #\n"
    "#     #  #   ###   ##   #####  ####   #       ##### ##### #  #   ### \n";

static void test_golden(void) {
    char buf[1024];
    TEST_ASSERT(tn_block_font_render("PROJECT ZERO", buf, sizeof(buf), 0) > 0, "renders");
    TEST_ASSERT(strcmp(buf, GOLDEN) == 0, "matches the brand reference ASCII exactly");
    char b2[1024];
    tn_block_font_render("project-zero", b2, sizeof(b2), 0);
    TEST_ASSERT(strcmp(b2, GOLDEN) == 0, "default --ui-title spells the same wordmark");
    tn_block_font_render("Project_Zero", b2, sizeof(b2), 0);
    TEST_ASSERT(strcmp(b2, GOLDEN) == 0, "case-insensitive, '_' as gap");
}

static void test_raw_and_limits(void) {
    char buf[256];
    TEST_ASSERT(tn_block_font_render("ZE", buf, sizeof(buf), 1) > 0, "raw render");
    TEST_ASSERT(strncmp(buf, "#####.#####\n", 12) == 0, "raw keeps '.' blanks and the 1-column gap");
    TEST_ASSERT(tn_block_font_render("PROJECT X", buf, sizeof(buf), 0) == 0, "unsupported letter -> 0");
    TEST_ASSERT(!tn_block_font_supports("HELLO"), "H/L unsupported");
    TEST_ASSERT(tn_block_font_supports("zero"), "supported subset");
    TEST_ASSERT(tn_block_font_render("", buf, sizeof(buf), 0) == 0, "empty -> 0");
    TEST_ASSERT(tn_block_font_render("PROJECT ZERO", buf, 40, 0) == 0, "too-small buffer -> 0, no overflow");
    TEST_ASSERT(buf[0] == '\0', "buffer left empty on failure");
}

static void test_banner_parity(void) {
    char a[TN_BANNER_PLAIN_BUF_CAP];
    tn_banner_format_plain(a, sizeof(a));
    TEST_ASSERT(strcmp(a, GOLDEN) == 0, "CLI banner == web logo (single font)");
}

int main(void) {
    printf("=== Block Font (brand wordmark) Tests ===\n");
    RUN_TEST(test_golden);
    RUN_TEST(test_raw_and_limits);
    RUN_TEST(test_banner_parity);
    TEST_SUMMARY();
}
