/*
 * tests/test_md_render.c
 *
 * Phase 22.3 — incremental markdown rendering, including constructs whose
 * delimiters are split across separate feed() calls (as real streamed
 * tokens would split them).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cli/md_render.h"

#define PASS(name)  printf("[PASS] %s\n", name)
#define FAIL(name, msg) do { printf("[FAIL] %s: %s\n", name, msg); g_failures++; } while(0)

static int g_failures = 0;

/* Feeds each piece in sequence, flushes, and returns the captured output
 * as a heap string the caller must free(). */
static char *render_pieces(const char *const *pieces, int n, int color_enabled) {
    char *out_buf = NULL;
    size_t out_size = 0;
    FILE *out = open_memstream(&out_buf, &out_size);

    MdRenderState st;
    md_render_init(&st, out, color_enabled);
    for (int i = 0; i < n; i++) md_render_feed(&st, pieces[i]);
    md_render_flush(&st);
    md_render_free(&st);

    fclose(out); /* finalizes out_buf/out_size */
    return out_buf;
}

static void test_plain_passthrough(void) {
    const char *pieces[] = {"Hello", " ", "world", "."};
    char *out = render_pieces(pieces, 4, 0);
    if (strcmp(out, "Hello world.") != 0) {
        FAIL("plain_passthrough", out);
    } else {
        PASS("plain_passthrough");
    }
    free(out);
}

static void test_bold_single_piece_no_color(void) {
    const char *pieces[] = {"This is **bold** text."};
    char *out = render_pieces(pieces, 1, 0);
    if (strcmp(out, "This is bold text.") != 0) {
        FAIL("bold_single_piece_no_color", out);
    } else {
        PASS("bold_single_piece_no_color");
    }
    free(out);
}

static void test_bold_split_across_tokens(void) {
    /* Deliberately split "**bold**" across many small token pieces,
     * including splitting the delimiter itself ("*" + "*"). */
    const char *pieces[] = {"go ", "*", "*", "bo", "ld", "*", "*", " now"};
    char *out = render_pieces(pieces, 8, 0);
    if (strcmp(out, "go bold now") != 0) {
        FAIL("bold_split_across_tokens", out);
    } else {
        PASS("bold_split_across_tokens");
    }
    free(out);
}

static void test_bold_with_color(void) {
    const char *pieces[] = {"**hi**"};
    char *out = render_pieces(pieces, 1, 1);
    if (!strstr(out, "hi")) { FAIL("bold_with_color", "missing content"); free(out); return; }
    if (!strstr(out, "\x1b[1m")) { FAIL("bold_with_color", "missing bold ANSI code"); free(out); return; }
    if (!strstr(out, "\x1b[0m")) { FAIL("bold_with_color", "missing reset ANSI code"); free(out); return; }
    if (strstr(out, "**")) { FAIL("bold_with_color", "markdown markers should be stripped"); free(out); return; }
    PASS("bold_with_color");
    free(out);
}

static void test_inline_code(void) {
    const char *pieces[] = {"run `make test` now"};
    char *out = render_pieces(pieces, 1, 0);
    if (strcmp(out, "run make test now") != 0) {
        FAIL("inline_code", out);
    } else {
        PASS("inline_code");
    }
    free(out);
}

static void test_fenced_code_block(void) {
    const char *pieces[] = {"before ```int x = 1;``` after"};
    char *out = render_pieces(pieces, 1, 0);
    if (strcmp(out, "before int x = 1; after") != 0) {
        FAIL("fenced_code_block", out);
    } else {
        PASS("fenced_code_block");
    }
    free(out);
}

static void test_single_backtick_not_confused_with_fence(void) {
    /* A lone inline-code span followed later by a real fence must not
     * mis-detect the first backtick as part of a fence. */
    const char *pieces[] = {"`x` and ```y```"};
    char *out = render_pieces(pieces, 1, 0);
    if (strcmp(out, "x and y") != 0) {
        FAIL("single_backtick_not_confused_with_fence", out);
    } else {
        PASS("single_backtick_not_confused_with_fence");
    }
    free(out);
}

static void test_unterminated_construct_flushed_as_raw(void) {
    /* "**never closes" has no closing "**" — flush must emit it verbatim,
     * not silently drop it. */
    const char *pieces[] = {"careful **never closes"};
    char *out = render_pieces(pieces, 1, 0);
    if (strcmp(out, "careful **never closes") != 0) {
        FAIL("unterminated_construct_flushed_as_raw", out);
    } else {
        PASS("unterminated_construct_flushed_as_raw");
    }
    free(out);
}

static void test_unclosed_fence_eventually_flushes_without_waiting_for_end(void) {
    /* An opening ``` that never closes must not block streaming forever —
     * once the unclosed span exceeds the safety threshold, it should flush
     * as plain text rather than waiting for md_render_flush(). */
    char big[5000];
    memset(big, 'x', sizeof(big) - 1);
    big[sizeof(big) - 1] = '\0';

    char *out_buf = NULL; size_t out_size = 0;
    FILE *out = open_memstream(&out_buf, &out_size);
    MdRenderState st;
    md_render_init(&st, out, 0);

    md_render_feed(&st, "```");
    md_render_feed(&st, big); /* pushes the unclosed span past the safety threshold */
    fflush(out);
    /* Without a call to md_render_flush(), the bulk of `big` should already
     * have been emitted once the safety valve tripped. */
    if (out_size < sizeof(big) / 2) {
        FAIL("unclosed_fence_eventually_flushes", "content did not stream before flush()");
    } else {
        PASS("unclosed_fence_eventually_flushes_without_waiting_for_end");
    }
    md_render_flush(&st);
    md_render_free(&st);
    fclose(out);
    free(out_buf);
}

static void test_empty_and_null_piece_no_crash(void) {
    MdRenderState st;
    char *out_buf = NULL; size_t out_size = 0;
    FILE *out = open_memstream(&out_buf, &out_size);
    md_render_init(&st, out, 0);
    md_render_feed(&st, "");
    md_render_feed(&st, NULL);
    md_render_feed(&st, "ok");
    md_render_flush(&st);
    md_render_free(&st);
    fclose(out);
    if (strcmp(out_buf, "ok") != 0) { FAIL("empty_and_null_piece_no_crash", out_buf); }
    else PASS("empty_and_null_piece_no_crash");
    free(out_buf);
}

int main(void) {
    printf("=== Phase 22.3 Markdown Render Tests ===\n");
    test_plain_passthrough();
    test_bold_single_piece_no_color();
    test_bold_split_across_tokens();
    test_bold_with_color();
    test_inline_code();
    test_fenced_code_block();
    test_single_backtick_not_confused_with_fence();
    test_unterminated_construct_flushed_as_raw();
    test_unclosed_fence_eventually_flushes_without_waiting_for_end();
    test_empty_and_null_piece_no_crash();

    printf("\n");
    if (g_failures == 0) { printf("=== All markdown render tests passed ===\n"); return 0; }
    printf("=== %d test(s) FAILED ===\n", g_failures);
    return 1;
}
