#include "test_harness.h"
#include "memory/mapped_file.h"
#include "core/platform.h"
#include <string.h>
#if TN_POSIX
#include <fcntl.h>
#include <unistd.h>
#endif

static void test_mmap_roundtrip(void) {
    /* Write a temp file */
    const char *tmppath = "/tmp/tn_test_mmap.bin";
    FILE *f = fopen(tmppath, "wb");
    TEST_ASSERT(f != NULL, "create temp file");

    tn_u8 data[256];
    for (int i = 0; i < 256; i++) data[i] = (tn_u8)i;
    fwrite(data, 1, 256, f);
    fclose(f);

    /* Map it */
    MappedFile mf;
    TernaryError err = mapped_file_open(&mf, tmppath);
    TEST_ASSERT_EQ(err, TN_OK, "mapped_file_open OK");
    TEST_ASSERT(mf.data != NULL, "data is non-NULL");
    TEST_ASSERT_EQ((int)mf.size, 256, "size is 256");

    /* Read back */
    tn_u8 *mapped = (tn_u8 *)mf.data;
    int match = 1;
    for (int i = 0; i < 256; i++) {
        if (mapped[i] != (tn_u8)i) { match = 0; break; }
    }
    TEST_ASSERT(match, "mapped data matches written data");

    /* Cleanup */
    mapped_file_close(&mf);
    TEST_ASSERT(mf.data == NULL, "data cleared after close");
    remove(tmppath);
}

static void test_mmap_nonexistent(void) {
    MappedFile mf;
    TernaryError err = mapped_file_open(&mf, "/tmp/tn_does_not_exist_12345.bin");
    TEST_ASSERT_EQ(err, TN_ERR_FILE_OPEN, "nonexistent file returns ERR_FILE_OPEN");
}

/* 2026-09-22: mapped_file_close()'s own doc comment promises it is "safe to
 * call on a zeroed MappedFile", but a plain memset-to-zero struct has
 * fd == 0 (not -1) until mapped_file_open() has run — closing it on a
 * never-opened struct used to hit `if (mf->fd >= 0) close(mf->fd)` and
 * silently close fd 0 (stdin). Found while adding a second caller
 * (cli/model_load.c's loaded_model_free()) that actually exercises this
 * exact zeroed-struct path. Verifies stdin's fd stays open and valid across
 * the call — the only way this bug can actually be observed from outside. */
static void test_mmap_close_on_zeroed_struct_does_not_close_stdin(void) {
#if TN_POSIX
    int before = fcntl(STDIN_FILENO, F_GETFD);
    TEST_ASSERT(before != -1, "stdin is open before the test (sanity check)");

    MappedFile mf;
    memset(&mf, 0, sizeof(mf));
    mapped_file_close(&mf);

    int after = fcntl(STDIN_FILENO, F_GETFD);
    TEST_ASSERT(after != -1, "stdin (fd 0) is still open after closing a zeroed MappedFile");
    TEST_ASSERT(mf.fd == -1, "zeroed MappedFile's fd is normalized to -1 after close");
#else
    TEST_ASSERT(1, "skipped on non-POSIX (Windows path already NULL-checks mf->handle)");
#endif
}

int main(void) {
    RUN_TEST(test_mmap_roundtrip);
    RUN_TEST(test_mmap_nonexistent);
    RUN_TEST(test_mmap_close_on_zeroed_struct_does_not_close_stdin);
    TEST_SUMMARY();
}
