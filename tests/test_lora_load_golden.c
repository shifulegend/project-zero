/*
 * tests/test_lora_load_golden.c — lora_load_ex over 3,000 generated adapter files: valid ones for
 * assorted target masks and shapes, ones built for a different base model, ones with an invalid
 * header field (each checked field out of range, a bad magic, a file under 64 bytes), and ones cut
 * short at a random byte. A hash of the result code and of everything loaded (header values, each
 * module's rank, scale and weights, which targets exist), recorded from the function before it was
 * split into steps.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "core/lora.h"
#include "test_harness.h"

#define EXPECTED_HASH 0x4c9186058e72f72bull
#define N_FILES 3000

static int g_loaded;
static uint64_t g_hash = 0xcbf29ce484222325ull;
static void mix(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        g_hash ^= b[i];
        g_hash *= 0x100000001b3ull;
    }
}

static uint64_t g_rng = 0x9e3779b97f4a7c15ull;
static unsigned rnd(unsigned n) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (unsigned)(g_rng % n);
}

typedef struct {
    int n_layers, rank, dim, hidden, kv, q;
    unsigned mask;
} Shape;

static void put32(uint8_t *buf, size_t off, uint32_t v) {
    memcpy(buf + off, &v, 4);
}

/* The (n, d) of each target for a model of this shape, in LoRaTargetBit order. */
static void target_dims(const Shape *s, int t, int *n, int *d) {
    switch (t) {
        case 0:
            *n = s->dim;
            *d = s->q;
            break;
        case 1:
        case 2:
            *n = s->dim;
            *d = s->kv;
            break;
        case 3:
            *n = s->q;
            *d = s->dim;
            break;
        case 4:
        case 5:
            *n = s->dim;
            *d = s->hidden;
            break;
        default:
            *n = s->hidden;
            *d = s->dim;
            break;
    }
}

static size_t payload_bytes(const Shape *s) {
    size_t total = 0;
    for (int t = 0; t < 7; t++) {
        if (!(s->mask & (1u << t))) continue;
        int n, d;
        target_dims(s, t, &n, &d);
        total += (size_t)s->n_layers * ((size_t)s->rank * n + (size_t)d * s->rank) * 2;
    }
    return total;
}

static void mix_module_array(const LoRAModule *arr, const Shape *s, int t) {
    int present = arr != NULL;
    mix(&present, sizeof present);
    if (!arr) return;
    int n, d;
    target_dims(s, t, &n, &d);
    for (int l = 0; l < s->n_layers; l++) {
        mix(&arr[l].rank, sizeof arr[l].rank);
        mix(&arr[l].scale, sizeof arr[l].scale);
        if (arr[l].A) mix(arr[l].A, (size_t)arr[l].rank * n * sizeof(float));
        if (arr[l].B) mix(arr[l].B, (size_t)d * arr[l].rank * sizeof(float));
    }
}

static void mix_lora(const LoRAWeights *w, const Shape *model) {
    mix(&w->n_layers, sizeof w->n_layers);
    mix(&w->rank, sizeof w->rank);
    mix(&w->alpha, sizeof w->alpha);
    mix(&w->dim, sizeof w->dim);
    mix(&w->hidden_dim, sizeof w->hidden_dim);
    mix_module_array(w->q, model, 0);
    mix_module_array(w->k, model, 1);
    mix_module_array(w->v, model, 2);
    mix_module_array(w->o, model, 3);
    mix_module_array(w->gate, model, 4);
    mix_module_array(w->up, model, 5);
    mix_module_array(w->down, model, 6);
}

/* The shape the file claims and the (sometimes different) model it is loaded into. */
static void gen_shapes(Shape *file, Shape *model) {
    file->n_layers = 1 + (int)rnd(3);
    file->rank     = 1 + (int)rnd(3);
    file->dim      = 4 + (int)rnd(5);
    file->kv       = 2 + (int)rnd(4);
    file->hidden   = (int)rnd(7);
    file->q        = rnd(3) ? file->dim : 2 + (int)rnd(8);
    file->mask     = rnd(256);
    *model         = *file;
    if (rnd(8) == 0) model->dim += 1 + (int)rnd(2); /* built for a different base model */
    if (rnd(8) == 0) model->n_layers += 1;
    if (rnd(8) == 0) model->kv += 1;
    if (rnd(8) == 0) model->hidden += 1;
    if (rnd(8) == 0) model->q += 1;
    if (rnd(5) == 0) model->q = 0; /* "q_dim <= 0 means dim" */
}

/* A valid file of the claimed shape with random payload bytes; *total is its length. */
static uint8_t *make_file(const Shape *file, size_t payload, size_t *total) {
    *total       = 64 + payload;
    uint8_t *buf = (uint8_t *)calloc(*total + 1, 1);
    for (size_t i = 64; i < *total; i++) buf[i] = (uint8_t)rnd(256);
    put32(buf, 0, 0x41524F4Cu);
    put32(buf, 8, (uint32_t)file->n_layers);
    put32(buf, 12, (uint32_t)file->rank);
    float alpha = 0.5f * (float)(1 + rnd(8));
    memcpy(buf + 16, &alpha, 4);
    put32(buf, 20, (uint32_t)file->dim);
    put32(buf, 24, (uint32_t)file->hidden);
    put32(buf, 28, file->mask);
    put32(buf, 32, (uint32_t)file->kv);
    put32(buf, 36, rnd(6) == 0 ? 0u : (uint32_t)file->q); /* 0 = pre-2026-10-19 file */
    return buf;
}

/* One random kind of damage (or none) to a header field or the file length. */
static void damage_file(uint8_t *buf, size_t payload, size_t *total) {
    static const struct {
        size_t off;
        uint32_t value;
    } k_fields[]    = {{0, 0xDEADBEEFu},  {8, 0u},  {8, 513u},    {12, 0u},
                       {12, 4097u},       {20, 0u}, {20, 65537u}, {24, 262145u},
                       {24, 0xFFFFFFFFu}, {32, 0u}, {32, 65537u}, {36, 262145u}};
    unsigned damage = rnd(20);
    if (damage < sizeof(k_fields) / sizeof(k_fields[0]))
        put32(buf, k_fields[damage].off, k_fields[damage].value);
    else if (damage == 12)
        *total = rnd(64); /* under 64 bytes */
    else if (damage == 13)
        *total = 64 + rnd((unsigned)payload + 1); /* cut short */
}

static void load_and_hash(const char *path, const Shape *file, const Shape *model) {
    LoRAWeights w;
    memset(&w, 0xA5, sizeof w); /* the loader must not depend on the incoming contents */
    int rc = (int)lora_load_ex(&w, path, model->dim, model->q, model->kv, model->hidden,
                               model->n_layers);
    mix(&rc, sizeof rc);
    if (rc == TN_OK) {
        g_loaded++;
        Shape eff = *model; /* what was loaded: the model's shape, q_dim <= 0 meaning dim */
        if (eff.q <= 0) eff.q = eff.dim;
        mix_lora(&w, &eff);
        lora_free(&w);
        return;
    }
    (void)file;
    const void *const arrays[] = {w.q, w.k, w.v, w.o, w.gate, w.up, w.down};
    int zeroed                 = 1;
    for (size_t i = 0; i < sizeof(arrays) / sizeof(arrays[0]); i++) zeroed &= arrays[i] == NULL;
    mix(&zeroed, sizeof zeroed);
}

static void one_case(const char *path) {
    Shape file, model;
    gen_shapes(&file, &model);
    size_t payload = payload_bytes(&file);
    size_t total;
    uint8_t *buf = make_file(&file, payload, &total);
    damage_file(buf, payload, &total);
    FILE *f = fopen(path, "wb");
    fwrite(buf, 1, total, f);
    fclose(f);
    free(buf);
    load_and_hash(path, &file, &model);
}

static void test_load_golden(void) {
    char path[] = "/tmp/pz_lora_golden_XXXXXX";
    int fd      = mkstemp(path);
    TEST_ASSERT(fd >= 0, "temp file");
    close(fd);
    for (int i = 0; i < N_FILES; i++) one_case(path);
    remove(path);
    if (getenv("LORA_GOLDEN_PRINT"))
        printf("loaded %d of %d, hash 0x%016llxull\n", g_loaded, N_FILES,
               (unsigned long long)g_hash);
    TEST_ASSERT(g_loaded > 250, "the generator reaches the success path");
    TEST_ASSERT(g_hash == EXPECTED_HASH, "loads match the recorded hash");
}

int main(void) {
    printf("=== lora_load_ex golden ===\n");
    RUN_TEST(test_load_golden);
    TEST_SUMMARY();
}
