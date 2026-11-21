/*
 * tests/test_gguf_config_golden.c — config_from_gguf and moe_config_from_gguf over 60,000
 * generated GGUF headers built in memory: every architecture the loader knows (llama, bitnet,
 * mamba, qwen3, qwen35, qwen3moe, deepseek2) and unknown ones, with each key the two functions
 * read present or absent and given a value of the right or a wrong type (unsigned, signed, float,
 * bool, string), zero, small, typical and huge values, YaRN scaling keys, and a token_embd.weight
 * tensor of one to three dimensions (or none). The hash covers both result codes and every byte of
 * both configs (pre-filled, so padding and untouched fields count; the YaRN attention factor is
 * rounded to four places because compilers differ in its last bit), recorded from the functions
 * before they were split into per-architecture steps.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/gguf_reader.h"
#include "runtime/gguf_loader.h"
#include "test_harness.h"

#define EXPECTED_HASH 0x38658ef00f4b3c74ull
#define N_CASES 60000

static uint64_t g_hash = 0xcbf29ce484222325ull;
static void mix(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        g_hash ^= b[i];
        g_hash *= 0x100000001b3ull;
    }
}

static uint64_t g_rng = 0x510e527fade682d1ull;
static unsigned rnd(unsigned n) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (unsigned)(g_rng % n);
}

static const char *const ARCHS[] = {"llama", "bitnet-b1.58", "bitnet",   "mamba",
                                    "qwen3", "qwen35",       "qwen3moe", "deepseek2",
                                    "phi3",  "gemma2",       "",         "qwen2"};
static const char *const KEYS[]  = {
    "embedding_length",
    "block_count",
    "attention.head_count",
    "feed_forward_length",
    "attention.head_count_kv",
    "context_length",
    "vocab_size",
    "rope.freq_base",
    "attention.layer_norm_rms_epsilon",
    "rope.scaling.factor",
    "rope.scaling.type",
    "rope.scaling.yarn_log_multiplier",
    "rope.scaling.original_context_length",
    "ssm.conv_kernel",
    "ssm.inner_size",
    "ssm.state_size",
    "ssm.time_step_rank",
    "ssm.group_count",
    "full_attention_interval",
    "attention.key_length",
    "rope.dimension_count",
    "expert_count",
    "expert_used_count",
    "expert_feed_forward_length",
    "leading_dense_block_count",
    "expert_shared_count",
    "attention.kv_lora_rank",
    "attention.value_length",
    "expert_gating_func",
    "expert_weights_norm",
    "expert_weights_scale",
};
static const char *const OTHER_KEYS[] = {"tokenizer.ggml.bos_token_id",
                                         "tokenizer.ggml.eos_token_id"};
static const uint32_t U32S[]          = {0,  1,   2,   3,   4,    6,    8,     16,     32,
                                         64, 128, 192, 512, 1408, 4096, 32768, 100000, 0xFFFFFFFFu};
static const float F32S[]             = {0.0f,  1.0f,  2.0f,  40.0f,    1e-6f,     1e-5f,
                                         1e-3f, 1e-4f, 1e-2f, 10000.0f, 500000.0f, -1.0f};
static const char *const STRS[]       = {"yarn", "linear", "none", "", "yarn-ext"};

static GGUFHeader *g_hdr;

static void set_u32(GGUFMeta *m) {
    m->val_type = GGUF_VAL_UINT32;
    m->val.u32  = U32S[rnd(sizeof U32S / sizeof U32S[0])];
}

static void set_f32(GGUFMeta *m) {
    m->val_type = GGUF_VAL_FLOAT32;
    m->val.f32  = F32S[rnd(sizeof F32S / sizeof F32S[0])];
}

static void set_i32(GGUFMeta *m) {
    uint32_t base = U32S[rnd(sizeof U32S / sizeof U32S[0])];
    int32_t shift = rnd(2) ? 100 : 0;
    m->val_type   = GGUF_VAL_INT32;
    m->val.i32    = (int32_t)base - shift;
}

static void set_bool_or_u64(GGUFMeta *m) {
    m->val_type = rnd(2) ? GGUF_VAL_BOOL : GGUF_VAL_UINT64;
    m->val.u64  = rnd(3);
}

static void set_string(GGUFMeta *m) {
    const char *s     = STRS[rnd(sizeof STRS / sizeof STRS[0])];
    m->val_type       = GGUF_VAL_STRING;
    m->val.string.str = (char *)s;
    m->val.string.len = strlen(s);
}

static void set_value(GGUFMeta *m) {
    switch (rnd(10)) {
        case 0:
        case 1:
        case 2:
        case 3:
        case 4:
            set_u32(m);
            break;
        case 5:
        case 6:
            set_f32(m);
            break;
        case 7:
            set_i32(m);
            break;
        case 8:
            set_bool_or_u64(m);
            break;
        default:
            set_string(m);
            break;
    }
}

static void add_key(GGUFHeader *h, const char *key) {
    GGUFMeta *m = &h->meta[h->n_meta++];
    memset(m, 0, sizeof *m);
    snprintf(m->key, sizeof m->key, "%s", key);
    set_value(m);
}

/* Mostly well-formed: a key most models need gets a plausible unsigned value. */
static void maybe_make_plausible(GGUFHeader *h, size_t k) {
    if (k >= 7 && (k < 13 || k > 28)) return;
    if (rnd(10) >= 8) return;
    GGUFMeta *m = &h->meta[h->n_meta - 1];
    m->val_type = GGUF_VAL_UINT32;
    m->val.u32  = U32S[1 + rnd(16)];
}

static void add_config_keys(GGUFHeader *h, const char *arch, unsigned density) {
    for (size_t k = 0; k < sizeof KEYS / sizeof KEYS[0]; k++) {
        int skip = k < 7 ? rnd(10) == 0 : rnd(10) >= density;
        if (skip) continue;
        char key[128];
        snprintf(key, sizeof key, "%s.%s", arch, KEYS[k]);
        add_key(h, key);
        maybe_make_plausible(h, k);
    }
}

static void add_embedding_tensor(GGUFHeader *h) {
    GGUFTensor *t = &h->tensors[h->n_tensors++];
    memset(t, 0, sizeof *t);
    snprintf(t->name, sizeof t->name, "token_embd.weight");
    t->n_dims = 1 + rnd(3);
    for (unsigned d = 0; d < 4; d++) {
        if (d < t->n_dims) t->dims[d] = U32S[rnd(12)];
    }
}

static void build(GGUFHeader *h) {
    memset(h, 0,
           4096); /* version, counts and the first entries; the rest is only read below n_meta */
    h->n_meta        = 0;
    h->n_tensors     = 0;
    const char *arch = ARCHS[rnd(sizeof ARCHS / sizeof ARCHS[0])];
    snprintf(h->arch, sizeof h->arch, "%s", arch);
    unsigned density = 3 + rnd(8); /* out of 10: how many of the keys are present */
    add_config_keys(h, arch, density);
    for (size_t k = 0; k < sizeof OTHER_KEYS / sizeof OTHER_KEYS[0]; k++)
        if (rnd(2)) add_key(h, OTHER_KEYS[k]);
    if (rnd(4) != 0) add_embedding_tensor(h);
}

static void test_gguf_config_golden(void) {
    g_hdr    = (GGUFHeader *)calloc(1, sizeof *g_hdr);
    int n_ok = 0, n_moe_ok = 0;
    for (int t = 0; t < N_CASES; t++) {
        build(g_hdr);
        Config cfg;
        MoEConfig mc;
        memset(&cfg, 0xAA, sizeof cfg);
        memset(&mc, 0xAA, sizeof mc);
        int rc1 = (int)config_from_gguf(&cfg, g_hdr);
        int rc2 = (int)moe_config_from_gguf(&mc, g_hdr);
        n_ok += rc1 == 0;
        n_moe_ok += rc2 == 0;
        /* logf / divide results differ in the last bit between compilers: pin them rounded */
        cfg.rope_yarn_attn_factor = roundf(cfg.rope_yarn_attn_factor * 1e4f) / 1e4f;
        mix(&rc1, sizeof rc1);
        mix(&cfg, sizeof cfg);
        mix(&rc2, sizeof rc2);
        mix(&mc, sizeof mc);
    }
    free(g_hdr);
    if (getenv("GGUF_CONFIG_GOLDEN_PRINT"))
        printf("hash 0x%016llxull (%d dense configs, %d moe configs accepted)\n",
               (unsigned long long)g_hash, n_ok, n_moe_ok);
    TEST_ASSERT(g_hash == EXPECTED_HASH, "configs match the recorded hash");
}

int main(void) {
    if (!getenv("GGUF_CONFIG_GOLDEN_VERBOSE")) freopen("/dev/null", "w", stderr);
    printf("=== gguf config golden ===\n");
    RUN_TEST(test_gguf_config_golden);
    TEST_SUMMARY();
}
