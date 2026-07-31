/*
 * tests/test_cancel.c
 *
 * Phase 22 — In-flight generation cancellation registry.
 */

#include <stdio.h>
#include "api/cancel.h"

#define PASS(name)  printf("[PASS] %s\n", name)
#define FAIL(name, msg) do { printf("[FAIL] %s: %s\n", name, msg); g_failures++; } while(0)

static int g_failures = 0;

static void test_no_active_generation(void) {
    CancelState cs;
    cancel_state_init(&cs);
    if (cancel_registry_request(&cs, "abc")) {
        FAIL("no_active_generation", "should not match with nothing active"); goto done;
    }
    if (cancel_registry_check(&cs, "abc")) {
        FAIL("no_active_generation", "check should be false with nothing active"); goto done;
    }
    PASS("no_active_generation");
done:
    cancel_state_destroy(&cs);
}

static void test_begin_request_check_lifecycle(void) {
    CancelState cs;
    cancel_state_init(&cs);
    cancel_registry_begin(&cs, "req-1");

    if (cancel_registry_check(&cs, "req-1")) {
        FAIL("begin_request_check_lifecycle", "should not be cancelled yet"); goto done;
    }
    if (!cancel_registry_request(&cs, "req-1")) {
        FAIL("begin_request_check_lifecycle", "request should match active id"); goto done;
    }
    if (!cancel_registry_check(&cs, "req-1")) {
        FAIL("begin_request_check_lifecycle", "check should now report cancelled"); goto done;
    }
    cancel_registry_end(&cs);
    if (cancel_registry_check(&cs, "req-1")) {
        FAIL("begin_request_check_lifecycle", "check must be false after end()"); goto done;
    }
    PASS("begin_request_check_lifecycle");
done:
    cancel_state_destroy(&cs);
}

static void test_id_mismatch_does_not_cancel(void) {
    CancelState cs;
    cancel_state_init(&cs);
    cancel_registry_begin(&cs, "req-A");

    if (cancel_registry_request(&cs, "req-B")) {
        FAIL("id_mismatch_does_not_cancel", "mismatched id must not match"); goto done;
    }
    if (cancel_registry_check(&cs, "req-A")) {
        FAIL("id_mismatch_does_not_cancel", "req-A must not be cancelled"); goto done;
    }
    PASS("id_mismatch_does_not_cancel");
done:
    cancel_registry_end(&cs);
    cancel_state_destroy(&cs);
}

static void test_second_begin_resets_flag(void) {
    CancelState cs;
    cancel_state_init(&cs);
    cancel_registry_begin(&cs, "req-1");
    cancel_registry_request(&cs, "req-1");
    cancel_registry_end(&cs);

    /* A fresh generation with a new id must not inherit a stale cancel flag. */
    cancel_registry_begin(&cs, "req-2");
    if (cancel_registry_check(&cs, "req-2")) {
        FAIL("second_begin_resets_flag", "new generation must start uncancelled"); goto done;
    }
    PASS("second_begin_resets_flag");
done:
    cancel_registry_end(&cs);
    cancel_state_destroy(&cs);
}

int main(void) {
    printf("=== Phase 22 Cancel Registry Tests ===\n");
    test_no_active_generation();
    test_begin_request_check_lifecycle();
    test_id_mismatch_does_not_cancel();
    test_second_begin_resets_flag();

    printf("\n");
    if (g_failures == 0) { printf("=== All cancel tests passed ===\n"); return 0; }
    printf("=== %d test(s) FAILED ===\n", g_failures);
    return 1;
}
