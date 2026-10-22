/*
 * tests/test_stop_seq.c — streaming stop-sequence matcher.
 */
#include "test_harness.h"
#include "transformer/stop_seq.h"

typedef struct { char out[512]; int calls; int stop_after; } Sink;

static int sink_emit(const char *text, void *ud) {
    Sink *s = (Sink *)ud;
    strncat(s->out, text, sizeof(s->out) - strlen(s->out) - 1);
    s->calls++;
    return s->stop_after > 0 && s->calls >= s->stop_after;
}

static int feed_all(StopMatcher *m, const char *const *pieces, int n, Sink *s) {
    for (int i = 0; i < n; i++)
        if (stop_matcher_feed(m, pieces[i], sink_emit, s)) return 1;
    stop_matcher_flush(m, sink_emit, s);
    return 0;
}

static void test_passthrough_without_stops(void) {
    StopMatcher m; stop_matcher_init(&m, NULL, 0);
    Sink s = {{0}, 0, 0};
    const char *p[] = {"Hello", ", ", "world"};
    TEST_ASSERT_EQ(feed_all(&m, p, 3, &s), 0, "never stops");
    TEST_ASSERT(strcmp(s.out, "Hello, world") == 0, "text forwarded unchanged");
    stop_matcher_free(&m);
}

static void test_stop_across_pieces(void) {
    const char *stops[] = {"\nUser:"};
    StopMatcher m; stop_matcher_init(&m, stops, 1);
    Sink s = {{0}, 0, 0};
    const char *p[] = {"Answer", " is 4.", "\nUs", "er:", " more"};
    TEST_ASSERT_EQ(feed_all(&m, p, 5, &s), 1, "stop detected across pieces");
    TEST_ASSERT(strcmp(s.out, "Answer is 4.") == 0, "stop text and later text not emitted");
    stop_matcher_free(&m);
}

static void test_false_prefix_released(void) {
    const char *stops[] = {"END"};
    StopMatcher m; stop_matcher_init(&m, stops, 1);
    Sink s = {{0}, 0, 0};
    const char *p[] = {"the EN", "D", "ING"};   /* "END" appears -> stop */
    TEST_ASSERT_EQ(feed_all(&m, p, 3, &s), 1, "stops on END");
    TEST_ASSERT(strcmp(s.out, "the ") == 0, "only text before stop");
    stop_matcher_free(&m);

    stop_matcher_init(&m, stops, 1);
    Sink t = {{0}, 0, 0};
    const char *q[] = {"the EN", "ergy"};       /* held "EN" released once disambiguated */
    TEST_ASSERT_EQ(feed_all(&m, q, 2, &t), 0, "no stop");
    TEST_ASSERT(strcmp(t.out, "the ENergy") == 0, "held prefix released intact");
    stop_matcher_free(&m);
}

static void test_flush_emits_held_tail(void) {
    const char *stops[] = {"###"};
    StopMatcher m; stop_matcher_init(&m, stops, 1);
    Sink s = {{0}, 0, 0};
    const char *p[] = {"done #"};
    TEST_ASSERT_EQ(feed_all(&m, p, 1, &s), 0, "no stop");
    TEST_ASSERT(strcmp(s.out, "done #") == 0, "flush emits the held '#'");
    stop_matcher_free(&m);
}

static void test_earliest_of_multiple(void) {
    const char *stops[] = {"zzz", "b", ""};
    StopMatcher m; stop_matcher_init(&m, stops, 3);
    TEST_ASSERT_EQ(m.n_seqs, 2, "empty stop ignored");
    Sink s = {{0}, 0, 0};
    const char *p[] = {"aab zzz"};
    TEST_ASSERT_EQ(feed_all(&m, p, 1, &s), 1, "stops");
    TEST_ASSERT(strcmp(s.out, "aa") == 0, "earliest match wins");
    stop_matcher_free(&m);
}

static void test_emit_requested_stop_propagates(void) {
    StopMatcher m; stop_matcher_init(&m, NULL, 0);
    Sink s = {{0}, 0, 1};
    TEST_ASSERT_EQ(stop_matcher_feed(&m, "x", sink_emit, &s), 1, "callback stop honored");
    stop_matcher_free(&m);
}

int main(void) {
    printf("=== Stop Sequence Tests ===\n");
    RUN_TEST(test_passthrough_without_stops);
    RUN_TEST(test_stop_across_pieces);
    RUN_TEST(test_false_prefix_released);
    RUN_TEST(test_flush_emits_held_tail);
    RUN_TEST(test_earliest_of_multiple);
    RUN_TEST(test_emit_requested_stop_propagates);
    TEST_SUMMARY();
}
