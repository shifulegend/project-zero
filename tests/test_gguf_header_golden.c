/*
 * tests/test_gguf_header_golden.c — gguf_read_header over generated GGUF images: valid headers of
 * versions 1 to 3 with random metadata (integers, floats, bools, strings, arrays of both) and
 * tensor descriptions (zero to four dimensions, assorted types and offsets, vision-looking names).
 *   - Golden: a hash of every field the reader fills for each intact image, recorded from the
 *     function before it was split into steps and before its bounds checks were added (an intact
 *     image never exercised them).
 *   - Robustness: every truncation of each image and many single-byte mutations are read from an
 *     exactly sized buffer (ASan catches any overread) and the header is freed afterwards, as
 *     callers do on failure. A cut inside the header must be rejected. The old reader read scalar
 *     metadata values past the end of the data, trusted a string length that could wrap the
 *     pointer, looped over a huge scalar-array count, and left a huge declared count behind for
 *     gguf_header_free to walk.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/gguf_reader.h"
#include "test_harness.h"

#define EXPECTED_HASH 0x07890c18b3c7b53cull
#define N_IMAGES 40
#define N_MUTATIONS 200
#define IMG_CAP 4096

static uint64_t g_hash = 0xcbf29ce484222325ull;
static void mix(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        g_hash ^= b[i];
        g_hash *= 0x100000001b3ull;
    }
}
static void mix_u64(uint64_t v) {
    mix(&v, sizeof v);
}

static uint64_t g_rng = 0x9e3779b97f4a7c15ull;
static unsigned rnd(unsigned n) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (unsigned)(g_rng % n);
}

static uint64_t g_rng2 = 0x2545f4914f6cdd1dull;
static unsigned rnd2(unsigned n) {
    g_rng2 ^= g_rng2 << 13;
    g_rng2 ^= g_rng2 >> 7;
    g_rng2 ^= g_rng2 << 17;
    return (unsigned)(g_rng2 % n);
}

typedef struct {
    uint8_t b[IMG_CAP];
    size_t n;
} Img;

static void put(Img *im, const void *p, size_t n) {
    if (im->n + n <= IMG_CAP) memcpy(im->b + im->n, p, n);
    im->n += n;
}
static void put_u32(Img *im, uint32_t v) {
    put(im, &v, 4);
}
static void put_u64(Img *im, uint64_t v) {
    put(im, &v, 8);
}
static void put_str(Img *im, const char *s) {
    put_u64(im, strlen(s));
    put(im, s, strlen(s));
}

static const char *const k_keys[] = {
    "general.architecture", "general.name", "llama.block_count", "tokenizer.ggml.model", "x",
    "llama.rope.freq_base", "k.k.k"};
static const char *const k_strs[]  = {"llama", "qwen35", "", "a longer string value", "deepseek2"};
static const char *const k_names[] = {"token_embd.weight",
                                      "blk.0.attn_q.weight",
                                      "output.weight",
                                      "v.blk.0.attn_q.weight",
                                      "mm.proj.weight",
                                      "a",
                                      ""};
static const int k_types[]         = {0, 1, 2, 8, 12, 14, 30, 42, 99};

static void put_value(Img *im, uint32_t t) {
    switch (t) {
        case 0:
        case 1:
        case 7:
            put_u32(im, 0x01020304u);
            im->n -= 3;
            break; /* 1 byte */
        case 2:
        case 3:
            put_u32(im, 0x0506u);
            im->n -= 2;
            break;
        case 8:
            put_str(im, k_strs[rnd(sizeof k_strs / sizeof *k_strs)]);
            break;
        case 10:
        case 11:
        case 12:
            put_u64(im, rnd(1000));
            break;
        default:
            put_u32(im, rnd(100000));
            break;
    }
}

static void put_meta(Img *im) {
    put_str(im, k_keys[rnd(sizeof k_keys / sizeof *k_keys)]);
    uint32_t t = rnd(13); /* the thirteen defined value types, 0..12 */
    if (t == 9) {
        uint32_t et = rnd(3) == 0 ? 8 : 4;
        put_u32(im, 9);
        put_u32(im, et);
        uint64_t cnt = rnd(4);
        put_u64(im, cnt);
        for (uint64_t i = 0; i < cnt; i++) put_value(im, et);
        return;
    }
    put_u32(im, t);
    put_value(im, t);
}

static void put_tensor(Img *im) {
    put_str(im, k_names[rnd(sizeof k_names / sizeof *k_names)]);
    uint32_t nd = rnd(5); /* the format allows at most four dimensions */
    put_u32(im, nd);
    for (uint32_t d = 0; d < nd; d++) put_u64(im, 1 + rnd(70));
    put_u32(im, (uint32_t)k_types[rnd(sizeof k_types / sizeof *k_types)]);
    put_u64(im, rnd(4) == 0 ? 100000 + rnd(9) : rnd(200));
}

/* Builds an image; returns the length of the header proper (before alignment padding and data). */
static size_t build(Img *im, uint32_t version) {
    im->n          = 0;
    uint32_t magic = 0x46554747u; /* "GGUF" */
    put_u32(im, magic);
    put_u32(im, version);
    uint32_t nt = rnd(5), nm = rnd(7);
    if (version == 1) {
        put_u32(im, nt);
        put_u32(im, nm);
    } else {
        put_u64(im, nt);
        put_u64(im, nm);
    }
    for (uint32_t i = 0; i < nm; i++) put_meta(im);
    for (uint32_t i = 0; i < nt; i++) put_tensor(im);
    size_t header_len = im->n;
    size_t align      = version >= 2 ? 32 : 1;
    while (im->n % align) put(im, "\0", 1);
    size_t data = rnd(300);
    for (size_t i = 0; i < data; i++) put_u32(im, rnd(256)), im->n -= 3;
    return header_len;
}

static void mix_meta(const GGUFMeta *m, const uint8_t *base) {
    mix(m->key, strlen(m->key) + 1);
    mix_u64((uint64_t)m->val_type);
    switch (m->val_type) {
        case GGUF_VAL_STRING:
            mix_u64(m->val.string.len);
            if (m->val.string.str) mix(m->val.string.str, strlen(m->val.string.str) + 1);
            break;
        case GGUF_VAL_ARRAY:
            mix_u64((uint64_t)m->val.array.elem_type);
            mix_u64(m->val.array.count);
            mix_u64(m->val.array.data ? (uint64_t)((const uint8_t *)m->val.array.data - base)
                                      : ~0ull);
            break;
        default:
            mix(&m->val, 8);
            break;
    }
}

static void mix_header(TernaryError rc, const GGUFHeader *h, const uint8_t *base) {
    mix_u64((uint64_t)rc);
    mix_u64(h->version);
    mix_u64(h->n_tensors);
    mix_u64(h->n_meta);
    mix(h->arch, strlen(h->arch) + 1);
    mix_u64((uint64_t)h->is_multimodal);
    mix_u64(h->data_region ? (uint64_t)((const uint8_t *)h->data_region - base) : ~0ull);
    mix_u64(h->data_region_size);
    for (uint64_t i = 0; i < h->n_meta && i < GGUF_MAX_META_KEYS; i++) mix_meta(&h->meta[i], base);
    for (uint64_t i = 0; i < h->n_tensors && i < GGUF_MAX_TENSORS; i++) {
        const GGUFTensor *t = &h->tensors[i];
        mix(t->name, strlen(t->name) + 1);
        mix_u64(t->n_dims);
        for (int d = 0; d < 4; d++) mix_u64(t->dims[d]);
        mix_u64((uint64_t)t->type);
        mix_u64(t->offset);
        mix_u64(t->data ? (uint64_t)((const uint8_t *)t->data - base) : ~0ull);
        mix_u64(t->size_bytes);
    }
}

/* One read from an exactly sized copy. Hashes the whole header when `hash` is set and the parse
 * succeeded. The header is freed after failures too (callers do: the unload path). Returns the
 * result code. */
static TernaryError read_image(const uint8_t *bytes, size_t n, int hash) {
    uint8_t *copy = (uint8_t *)malloc(n ? n : 1); /* exact size: ASan sees any overread */
    memcpy(copy, bytes, n);
    GGUFHeader *h = (GGUFHeader *)malloc(sizeof(*h));
    memset(h, 0x5A, sizeof(*h)); /* the reader must not depend on the incoming contents */
    TernaryError rc = gguf_read_header(h, copy, n);
    if (hash && rc == TN_OK) mix_header(rc, h, copy);
    if (n >= 24) gguf_header_free(h); /* inputs under 24 bytes are rejected before h is touched */
    free(h);
    free(copy);
    return rc;
}

/* Truncations and mutations of one image, read for robustness only. Returns how many cuts inside
 * the header were wrongly accepted. */
static long hammer(const Img *im, size_t header_len, long *calls) {
    long bad_accept = 0;
    for (size_t cut = 0; cut < im->n; cut++, (*calls)++)
        if (read_image(im->b, cut, 0) == TN_OK && cut < header_len) bad_accept++;
    for (int m = 0; m < N_MUTATIONS; m++, (*calls)++) {
        uint8_t mutated[IMG_CAP];
        memcpy(mutated, im->b, im->n);
        mutated[rnd2((unsigned)im->n)] = (uint8_t)rnd2(256);
        (void)read_image(mutated, im->n, 0);
    }
    return bad_accept;
}

static void test_header_golden(void) {
    static Img imgs[N_IMAGES];
    size_t header_len[N_IMAGES];
    for (int i = 0; i < N_IMAGES; i++) {
        header_len[i] = build(&imgs[i], 1 + rnd(3));
        TEST_ASSERT(imgs[i].n <= IMG_CAP, "image fits the buffer");
        TEST_ASSERT(read_image(imgs[i].b, imgs[i].n, 1) == TN_OK, "an intact image parses");
    }
    if (getenv("GGUF_GOLDEN_PRINT")) printf("hash 0x%016llxull\n", (unsigned long long)g_hash);
    TEST_ASSERT(g_hash == EXPECTED_HASH, "intact header parses match the recorded hash");
    if (getenv("GGUF_GOLDEN_INTACT_ONLY")) return; /* recording from the pre-fix reader */

    long calls = 0, bad_accept = 0;
    for (int i = 0; i < N_IMAGES; i++) bad_accept += hammer(&imgs[i], header_len[i], &calls);
    TEST_ASSERT(calls > 3000, "enough cases");
    TEST_ASSERT(bad_accept == 0, "a cut inside the header is rejected");
}

int main(void) {
    printf("=== gguf_read_header golden ===\n");
    RUN_TEST(test_header_golden);
    TEST_SUMMARY();
}
