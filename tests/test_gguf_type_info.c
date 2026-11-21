/* gguf_type_* lookups: every tensor type keeps the name, block size and block element count it had
 * before the three per-type switches became one table (values captured from the old code for
 * types 0..70). */
#include <string.h>

#include "core/gguf_reader.h"
#include "test_harness.h"

typedef struct {
    int type;
    size_t bytes, elems;
    const char *name;
} Row;

static const Row k_rows[] = {
    {0, 4, 1, "F32"},         {1, 2, 1, "F16"},         {2, 18, 32, "Q4_0"},
    {3, 20, 32, "Q4_1"},      {6, 22, 32, "Q5_0"},      {7, 24, 32, "Q5_1"},
    {8, 34, 32, "Q8_0"},      {9, 36, 32, "Q8_1"},      {10, 84, 256, "Q2_K"},
    {11, 110, 256, "Q3_K"},   {12, 144, 256, "Q4_K"},   {13, 176, 256, "Q5_K"},
    {14, 210, 256, "Q6_K"},   {15, 292, 256, "Q8_K"},   {16, 66, 256, "IQ2_XXS"},
    {17, 74, 256, "IQ2_XS"},  {18, 98, 256, "IQ3_XXS"}, {19, 50, 256, "IQ1_S"},
    {20, 18, 32, "IQ4_NL"},   {21, 110, 256, "IQ3_S"},  {22, 82, 256, "IQ2_S"},
    {23, 136, 256, "IQ4_XS"}, {24, 1, 1, "I8"},         {25, 2, 1, "I16"},
    {26, 4, 1, "I32"},        {27, 8, 1, "I64"},        {28, 8, 1, "F64"},
    {29, 56, 256, "IQ1_M"},   {30, 2, 1, "BF16"},       {36, 0, 1, "I2_S"},
    {42, 34, 128, "Q2_0"}};

static void test_listed_types(void) {
    for (size_t i = 0; i < sizeof(k_rows) / sizeof(k_rows[0]); i++) {
        const Row *r = &k_rows[i];
        TEST_ASSERT(gguf_type_block_bytes((GGUFType)r->type) == r->bytes, "block bytes");
        TEST_ASSERT(gguf_type_block_elems((GGUFType)r->type) == r->elems, "block elements");
        TEST_ASSERT(strcmp(gguf_type_name((GGUFType)r->type), r->name) == 0, "name");
    }
}

static void test_unlisted_types(void) {
    for (int t = 0; t <= 70; t++) {
        int listed = 0;
        for (size_t i = 0; i < sizeof(k_rows) / sizeof(k_rows[0]); i++)
            listed |= k_rows[i].type == t;
        if (listed) continue;
        TEST_ASSERT(gguf_type_block_bytes((GGUFType)t) == 0, "unlisted type has no block size");
        TEST_ASSERT(gguf_type_block_elems((GGUFType)t) == 1, "and one element per block");
        TEST_ASSERT(strcmp(gguf_type_name((GGUFType)t), "UNKNOWN") == 0, "and the name UNKNOWN");
    }
}

static void test_bytes_per_element(void) {
    float bpe = gguf_type_bpe(GGUF_TYPE_Q4_K); /* 144 bytes per 256 weights */
    TEST_ASSERT(bpe > 0.5624f && bpe < 0.5626f, "Q4_K is 0.5625 bytes per weight");
}

int main(void) {
    RUN_TEST(test_listed_types);
    RUN_TEST(test_unlisted_types);
    RUN_TEST(test_bytes_per_element);
    TEST_SUMMARY();
}
