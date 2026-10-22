/*
 * test_gguf_geometry.c — TS-1.2: block-size/geometry assertions.
 *
 * Verifies gguf_reader.c's internal block-size table against the REAL
 * byte-parsing path (not just the formula in isolation): builds a minimal,
 * valid GGUF v3 binary buffer with one tensor of each type at exactly one
 * block's worth of elements, runs it through gguf_read_header(), and checks
 * the resulting t->size_bytes. This exercises tensor_size_bytes() ->
 * gguf_block_size()/gguf_block_elems() for real, not a reimplementation of
 * the formula.
 *
 * Also cross-checks the public gguf_type_bpe() (bytes-per-element) against
 * the same expected byte counts, so the two internal tables (block_size,
 * block_elems) that only gguf_type_bpe() exposes indirectly are covered
 * from both angles.
 */
#include "core/gguf_reader.h"
#include "test_harness.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ── Minimal GGUF v3 buffer builder: magic+version+counts, zero KV entries,
 * one tensor descriptor (name, n_dims=1, dims[0]=n_elems, type, offset=0),
 * padded to 32-byte alignment, then `data_bytes` of dummy payload. ── */
static uint8_t *build_minimal_gguf(GGUFType type, uint64_t n_elems,
                                    size_t data_bytes, size_t *out_size) {
    const char *tname = "t";
    size_t header_size = 4 + 4 + 8 + 8;                    /* magic+version+n_tensors+n_kv */
    size_t tensor_desc_size = 8 + strlen(tname) + 4 + 8 + 4 + 8; /* name_len+name+n_dims+dims[0]+type+offset */
    size_t pre_data = header_size + tensor_desc_size;
    size_t data_start = (pre_data + 31) & ~(size_t)31;
    size_t total = data_start + data_bytes;

    uint8_t *buf = (uint8_t *)calloc(1, total);
    if (!buf) return NULL;
    uint8_t *p = buf;

    uint32_t magic = 0x46554747u; memcpy(p, &magic, 4); p += 4;
    uint32_t version = 3; memcpy(p, &version, 4); p += 4;
    uint64_t n_tensors = 1; memcpy(p, &n_tensors, 8); p += 8;
    uint64_t n_kv = 0; memcpy(p, &n_kv, 8); p += 8;

    uint64_t name_len = strlen(tname);
    memcpy(p, &name_len, 8); p += 8;
    memcpy(p, tname, name_len); p += name_len;
    uint32_t n_dims = 1; memcpy(p, &n_dims, 4); p += 4;
    memcpy(p, &n_elems, 8); p += 8;
    uint32_t ttype = (uint32_t)type; memcpy(p, &ttype, 4); p += 4;
    uint64_t offset = 0; memcpy(p, &offset, 8); p += 8;

    *out_size = total;
    return buf;
}

typedef struct {
    GGUFType type;
    const char *name;
    uint64_t block_elems;
    size_t   expected_bytes;
} GeometryCase;

static const GeometryCase cases[] = {
    { GGUF_TYPE_F32,     "F32",     1,   4   },
    { GGUF_TYPE_F16,     "F16",     1,   2   },
    { GGUF_TYPE_BF16,    "BF16",    1,   2   },
    { GGUF_TYPE_Q4_0,    "Q4_0",    32,  18  },
    { GGUF_TYPE_Q4_1,    "Q4_1",    32,  20  },
    { GGUF_TYPE_Q5_0,    "Q5_0",    32,  22  },
    { GGUF_TYPE_Q5_1,    "Q5_1",    32,  24  },
    { GGUF_TYPE_Q8_0,    "Q8_0",    32,  34  },
    { GGUF_TYPE_Q8_1,    "Q8_1",    32,  36  }, /* the 40->36 bug fix */
    { GGUF_TYPE_Q2_K,    "Q2_K",    256, 84  },
    { GGUF_TYPE_Q3_K,    "Q3_K",    256, 110 },
    { GGUF_TYPE_Q4_K,    "Q4_K",    256, 144 },
    { GGUF_TYPE_Q5_K,    "Q5_K",    256, 176 },
    { GGUF_TYPE_Q6_K,    "Q6_K",    256, 210 },
    { GGUF_TYPE_Q8_K,    "Q8_K",    256, 292 },
    { GGUF_TYPE_IQ2_XXS, "IQ2_XXS", 256, 66  },
    { GGUF_TYPE_IQ2_XS,  "IQ2_XS",  256, 74  },
    { GGUF_TYPE_IQ2_S,   "IQ2_S",   256, 82  },
    { GGUF_TYPE_IQ3_XXS, "IQ3_XXS", 256, 98  },
    { GGUF_TYPE_IQ3_S,   "IQ3_S",   256, 110 },
    { GGUF_TYPE_IQ1_S,   "IQ1_S",   256, 50  }, /* plan doc said 26 -- wrong */
    { GGUF_TYPE_IQ1_M,   "IQ1_M",   256, 56  }, /* plan doc said 37 -- wrong */
    { GGUF_TYPE_IQ4_NL,  "IQ4_NL",  32,  18  },
    { GGUF_TYPE_IQ4_XS,  "IQ4_XS",  256, 136 },
    { GGUF_TYPE_Q2_0,    "Q2_0",    128, 34  },
};
#define N_CASES (sizeof(cases) / sizeof(cases[0]))

static void test_size_bytes_one_block_each_type(void) {
    for (size_t i = 0; i < N_CASES; i++) {
        const GeometryCase *c = &cases[i];
        size_t total;
        /* one full block's worth of dummy payload, generous padding */
        uint8_t *buf = build_minimal_gguf(c->type, c->block_elems,
                                           c->expected_bytes + 64, &total);
        TEST_ASSERT(buf != NULL, "buffer allocated");
        if (!buf) continue;

        GGUFHeader hdr;
        TernaryError err = gguf_read_header(&hdr, buf, total);
        TEST_ASSERT(err == TN_OK, "header parses for this type");

        const GGUFTensor *t = gguf_find_tensor(&hdr, "t");
        TEST_ASSERT(t != NULL, "tensor found");
        if (t) {
            char msg[128];
            snprintf(msg, sizeof(msg), "%s: size_bytes == %zu (block-size table)",
                     c->name, c->expected_bytes);
            TEST_ASSERT(t->size_bytes == c->expected_bytes, msg);
        }
        gguf_header_free(&hdr);
        free(buf);
    }
}

static void test_type_bpe_matches_expected(void) {
    for (size_t i = 0; i < N_CASES; i++) {
        const GeometryCase *c = &cases[i];
        float expected_bpe = (float)c->expected_bytes / (float)c->block_elems;
        float actual_bpe = gguf_type_bpe(c->type);
        char msg[128];
        snprintf(msg, sizeof(msg), "%s: gguf_type_bpe ~= %.4f", c->name, expected_bpe);
        TEST_ASSERT(actual_bpe > expected_bpe - 0.001f && actual_bpe < expected_bpe + 0.001f, msg);
    }
}

static void test_multi_block_scales_linearly(void) {
    /* 4 blocks of Q4_K should be exactly 4x the single-block byte count,
     * via the real parsing path (not just the formula). */
    size_t total;
    uint8_t *buf = build_minimal_gguf(GGUF_TYPE_Q4_K, 256 * 4, 144 * 4 + 64, &total);
    TEST_ASSERT(buf != NULL, "buffer allocated");
    if (buf) {
        GGUFHeader hdr;
        TernaryError err = gguf_read_header(&hdr, buf, total);
        TEST_ASSERT(err == TN_OK, "4-block header parses");
        const GGUFTensor *t = gguf_find_tensor(&hdr, "t");
        TEST_ASSERT(t && t->size_bytes == 144 * 4, "4 blocks of Q4_K == 4x single-block size");
        gguf_header_free(&hdr);
        free(buf);
    }
}

static void test_type_name_no_unknowns(void) {
    /* Every type in our table must resolve to a real name, not "UNKNOWN" --
     * catches a type added to the enum but forgotten in gguf_type_name(). */
    for (size_t i = 0; i < N_CASES; i++) {
        const char *n = gguf_type_name(cases[i].type);
        char msg[128];
        snprintf(msg, sizeof(msg), "%s: gguf_type_name is not UNKNOWN", cases[i].name);
        TEST_ASSERT(strcmp(n, "UNKNOWN") != 0, msg);
    }
}

int main(void) {
    RUN_TEST(test_size_bytes_one_block_each_type);
    RUN_TEST(test_type_bpe_matches_expected);
    RUN_TEST(test_multi_block_scales_linearly);
    RUN_TEST(test_type_name_no_unknowns);
    TEST_SUMMARY();
}
