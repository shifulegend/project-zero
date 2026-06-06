#include "test_harness.h"
#include "memory/mapped_file.h"
#include "core/platform.h"

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

int main(void) {
    RUN_TEST(test_mmap_roundtrip);
    RUN_TEST(test_mmap_nonexistent);
    TEST_SUMMARY();
}
