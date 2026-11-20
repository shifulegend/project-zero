#include "core/gguf_reader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/log.h"

/* ── Internal helpers ───────────────────────────────────────────────────── */

/* Read a little-endian uint64 from a byte cursor, advancing it. */
static uint64_t read_u64(const uint8_t **p) {
    uint64_t v;
    memcpy(&v, *p, 8);
    *p += 8;
    return v;
}

static uint32_t read_u32(const uint8_t **p) {
    uint32_t v;
    memcpy(&v, *p, 4);
    *p += 4;
    return v;
}

static uint16_t read_u16(const uint8_t **p) {
    uint16_t v;
    memcpy(&v, *p, 2);
    *p += 2;
    return v;
}

static uint8_t read_u8(const uint8_t **p) {
    return *(*p)++;
}

/* Read a GGUF string (uint64 len + bytes, no NUL in file).
 * Always advances *p by the full on-disk length even if truncating to out_sz. */
static int read_str(const uint8_t **p, const uint8_t *end, char *out, size_t out_sz) {
    if (*p + 8 > end) return -1;
    uint64_t len = read_u64(p);
    if (*p + len > end) return -1;
    size_t copy_len = (len < out_sz - 1) ? (size_t)len : out_sz - 1;
    memcpy(out, *p, copy_len);
    out[copy_len] = '\0';
    *p += len; /* advance by full on-disk length, not truncated copy */
    return 0;
}

/* Skip a metadata value of any type (used to skip unknown/array entries). */
static int skip_meta_value(const uint8_t **p, const uint8_t *end, GGUFValType vtype);

static int skip_meta_value(const uint8_t **p, const uint8_t *end, GGUFValType vtype) {
    if (*p >= end) return -1;
    switch (vtype) {
        case GGUF_VAL_UINT8:
        case GGUF_VAL_INT8:
        case GGUF_VAL_BOOL:
            *p += 1;
            break;
        case GGUF_VAL_UINT16:
        case GGUF_VAL_INT16:
            *p += 2;
            break;
        case GGUF_VAL_UINT32:
        case GGUF_VAL_INT32:
        case GGUF_VAL_FLOAT32:
            *p += 4;
            break;
        case GGUF_VAL_UINT64:
        case GGUF_VAL_INT64:
        case GGUF_VAL_FLOAT64:
            *p += 8;
            break;
        case GGUF_VAL_STRING: {
            /* Skip string bytes without copying — just advance past them. */
            if (*p + 8 > end) return -1;
            uint64_t slen = read_u64(p);
            if (*p + slen > end) return -1;
            *p += slen;
            break;
        }
        case GGUF_VAL_ARRAY: {
            if (*p + 12 > end) return -1;
            GGUFValType elem_type = (GGUFValType)read_u32(p);
            uint64_t count        = read_u64(p);
            for (uint64_t i = 0; i < count; i++) {
                if (skip_meta_value(p, end, elem_type)) return -1;
            }
            break;
        }
        default:
            return -1;
    }
    return 0;
}

/* Parse one metadata entry into *m. Returns 0 on success. */
static int parse_meta_entry(const uint8_t **p, const uint8_t *end, GGUFMeta *m) {
    if (read_str(p, end, m->key, sizeof(m->key))) return -1;
    if (*p + 4 > end) return -1;
    m->val_type = (GGUFValType)read_u32(p);

    switch (m->val_type) {
        case GGUF_VAL_UINT8:
            m->val.u8 = read_u8(p);
            break;
        case GGUF_VAL_INT8:
            m->val.i8 = (int8_t)read_u8(p);
            break;
        case GGUF_VAL_UINT16:
            m->val.u16 = read_u16(p);
            break;
        case GGUF_VAL_INT16:
            m->val.i16 = (int16_t)read_u16(p);
            break;
        case GGUF_VAL_UINT32:
            m->val.u32 = read_u32(p);
            break;
        case GGUF_VAL_INT32:
            m->val.i32 = (int32_t)read_u32(p);
            break;
        case GGUF_VAL_FLOAT32:
            memcpy(&m->val.f32, *p, 4);
            *p += 4;
            break;
        case GGUF_VAL_BOOL:
            m->val.boolean = read_u8(p);
            break;
        case GGUF_VAL_UINT64:
            m->val.u64 = read_u64(p);
            break;
        case GGUF_VAL_INT64:
            m->val.i64 = (int64_t)read_u64(p);
            break;
        case GGUF_VAL_FLOAT64:
            memcpy(&m->val.f64, *p, 8);
            *p += 8;
            break;
        case GGUF_VAL_STRING: {
            if (*p + 8 > end) return -1;
            uint64_t slen = read_u64(p);
            if (*p + slen > end) return -1;
            m->val.string.len = slen;
            /* Heap-allocate a NUL-terminated copy so callers can safely use
             * strcmp / printf / strdup without reading past the string end.
             * GGUF on-disk strings have no NUL; the raw mmap pointer is unsafe. */
            m->val.string.str = (char *)malloc(slen + 1);
            if (!m->val.string.str) return -1;
            memcpy(m->val.string.str, *p, slen);
            m->val.string.str[slen] = '\0';
            *p += slen;
            break;
        }
        case GGUF_VAL_ARRAY: {
            if (*p + 12 > end) return -1;
            GGUFValType elem_type  = (GGUFValType)read_u32(p);
            uint64_t count         = read_u64(p);
            m->val.array.elem_type = elem_type;
            m->val.array.count     = count;
            m->val.array.data      = *p; /* zero-copy pointer into mmap */
            /* Advance cursor past all array elements */
            for (uint64_t i = 0; i < count; i++) {
                if (skip_meta_value(p, end, elem_type)) return -1;
            }
            break;
        }
        default:
            return -1;
    }
    return 0;
}

/* ── What every GGUF tensor type is: name, bytes per block and weights per block ───────────────
 * One row per type (a new quantisation is one row). Block byte counts were verified 2026-09-17
 * against llama.cpp's block_* structs in ggml-common.h (Phase 37): Q8_1 is 36 (d+s as an fp16 pair
 * plus 32 int8, no qh), IQ1_S is 50 and IQ1_M is 56. Q2_0 is PrismML's group-128 packing
 * (34 bytes per 128 weights, informational: t->data is located by the stored offset). I2_S is not
 * block-structured (size handled in tensor_size_bytes). A type that is not listed has size 0. */
typedef struct {
    GGUFType type;
    const char *name;
    size_t block_bytes;
    size_t block_elems;
} GGUFTypeInfo;

static const GGUFTypeInfo k_type_info[] = {
    {GGUF_TYPE_F32, "F32", 4, 1},
    {GGUF_TYPE_F16, "F16", 2, 1},
    {GGUF_TYPE_Q4_0, "Q4_0", 18, 32},
    {GGUF_TYPE_Q4_1, "Q4_1", 20, 32},
    {GGUF_TYPE_Q5_0, "Q5_0", 22, 32},
    {GGUF_TYPE_Q5_1, "Q5_1", 24, 32},
    {GGUF_TYPE_Q8_0, "Q8_0", 34, 32},
    {GGUF_TYPE_Q8_1, "Q8_1", 36, 32},
    {GGUF_TYPE_Q2_K, "Q2_K", 84, 256},
    {GGUF_TYPE_Q3_K, "Q3_K", 110, 256},
    {GGUF_TYPE_Q4_K, "Q4_K", 144, 256},
    {GGUF_TYPE_Q5_K, "Q5_K", 176, 256},
    {GGUF_TYPE_Q6_K, "Q6_K", 210, 256},
    {GGUF_TYPE_Q8_K, "Q8_K", 292, 256},
    {GGUF_TYPE_IQ2_XXS, "IQ2_XXS", 66, 256},
    {GGUF_TYPE_IQ2_XS, "IQ2_XS", 74, 256},
    {GGUF_TYPE_IQ3_XXS, "IQ3_XXS", 98, 256},
    {GGUF_TYPE_IQ1_S, "IQ1_S", 50, 256},
    {GGUF_TYPE_IQ4_NL, "IQ4_NL", 18, 32},
    {GGUF_TYPE_IQ3_S, "IQ3_S", 110, 256},
    {GGUF_TYPE_IQ2_S, "IQ2_S", 82, 256},
    {GGUF_TYPE_IQ4_XS, "IQ4_XS", 136, 256},
    {GGUF_TYPE_I8, "I8", 1, 1},
    {GGUF_TYPE_I16, "I16", 2, 1},
    {GGUF_TYPE_I32, "I32", 4, 1},
    {GGUF_TYPE_I64, "I64", 8, 1},
    {GGUF_TYPE_F64, "F64", 8, 1},
    {GGUF_TYPE_IQ1_M, "IQ1_M", 56, 256},
    {GGUF_TYPE_BF16, "BF16", 2, 1},
    {GGUF_TYPE_I2_S, "I2_S", 0, 1},
    {GGUF_TYPE_Q2_0, "Q2_0", 34, 128},
};

static const GGUFTypeInfo *type_info(GGUFType t) {
    for (size_t i = 0; i < sizeof(k_type_info) / sizeof(k_type_info[0]); i++)
        if (k_type_info[i].type == t) return &k_type_info[i];
    return NULL;
}

size_t gguf_type_block_bytes(GGUFType t) {
    const GGUFTypeInfo *info = type_info(t);
    return info ? info->block_bytes : 0;
}

size_t gguf_type_block_elems(GGUFType t) {
    const GGUFTypeInfo *info = type_info(t);
    return info ? info->block_elems : 1;
}

const char *gguf_type_name(GGUFType t) {
    const GGUFTypeInfo *info = type_info(t);
    return info ? info->name : "UNKNOWN";
}

static size_t tensor_size_bytes(const GGUFTensor *t) {
    uint64_t n_elems = 1;
    for (uint32_t i = 0; i < t->n_dims; i++) n_elems *= t->dims[i];
    /* I2_S is not block-structured: n/4 code bytes + F32 scale, padded to 32
     * (ggml_nbytes in bitnet.cpp's ggml.c). */
    if (t->type == GGUF_TYPE_I2_S) return (size_t)(n_elems / 4) + 32;
    size_t block_sz = gguf_type_block_bytes(t->type);
    size_t block_el = gguf_type_block_elems(t->type);
    size_t n_blocks = (size_t)((n_elems + block_el - 1) / block_el);
    return n_blocks * block_sz;
}

/* ── Vision tensor name patterns ────────────────────────────────────────── */
static const char *const VISION_PREFIXES[] = {
    "v.blk.", "vision_model.", "vision_tower.", "encoder.", "mm.", "vlm.", "image_encoder.", NULL};

static int looks_like_vision_tensor(const char *name) {
    for (int i = 0; VISION_PREFIXES[i]; i++) {
        if (strncmp(name, VISION_PREFIXES[i], strlen(VISION_PREFIXES[i])) == 0) return 1;
    }
    return 0;
}

/* ── Public API ─────────────────────────────────────────────────────────── */

TernaryError gguf_read_header(GGUFHeader *hdr, const void *mapped_ptr, size_t mapped_size) {
    if (!hdr || !mapped_ptr || mapped_size < 24) return TN_ERR_INVALID_ARGS;

    memset(hdr, 0, sizeof(*hdr));

    const uint8_t *p   = (const uint8_t *)mapped_ptr;
    const uint8_t *end = p + mapped_size;

    /* Magic */
    uint32_t magic = read_u32(&p);
    if (magic != GGUF_MAGIC) {
        TN_LOG_INF("gguf", "[gguf] bad magic: 0x%08X (expected 0x%08X)\n", magic, GGUF_MAGIC);
        return TN_ERR_INVALID_ARGS;
    }

    hdr->version = read_u32(&p);
    if (hdr->version < 1 || hdr->version > 3) {
        TN_LOG_ERR("gguf", "[gguf] unsupported version: %u\n", hdr->version);
        return TN_ERR_INVALID_ARGS;
    }

    hdr->n_tensors = (hdr->version == 1) ? read_u32(&p) : read_u64(&p);
    hdr->n_meta    = (hdr->version == 1) ? read_u32(&p) : read_u64(&p);

    if (hdr->n_tensors > GGUF_MAX_TENSORS || hdr->n_meta > GGUF_MAX_META_KEYS) {
        TN_LOG_INF("gguf", "[gguf] tensor count %llu or meta count %llu exceeds limit\n",
                   (unsigned long long)hdr->n_tensors, (unsigned long long)hdr->n_meta);
        return TN_ERR_INVALID_ARGS;
    }

    /* Parse metadata */
    for (uint64_t i = 0; i < hdr->n_meta; i++) {
        GGUFMeta *m = &hdr->meta[i];
        if (parse_meta_entry(&p, end, m) != 0) {
            TN_LOG_ERR("gguf", "[gguf] failed parsing meta entry %llu\n", (unsigned long long)i);
            return TN_ERR_INVALID_ARGS;
        }

        /* Extract convenience fields */
        if (strcmp(m->key, "general.architecture") == 0 && m->val_type == GGUF_VAL_STRING) {
            size_t n = m->val.string.len < sizeof(hdr->arch) - 1 ? m->val.string.len
                                                                 : sizeof(hdr->arch) - 1;
            memcpy(hdr->arch, m->val.string.str, n);
            hdr->arch[n] = '\0';
        }
    }

    /* Parse tensor info */
    for (uint64_t i = 0; i < hdr->n_tensors; i++) {
        GGUFTensor *t = &hdr->tensors[i];
        if (read_str(&p, end, t->name, sizeof(t->name))) {
            TN_LOG_ERR("gguf", "[gguf] failed reading tensor %llu name\n", (unsigned long long)i);
            return TN_ERR_INVALID_ARGS;
        }
        if (p + 4 > end) return TN_ERR_INVALID_ARGS;
        t->n_dims = read_u32(&p);
        if (t->n_dims > 4) {
            t->n_dims = 4;
        } /* cap at 4 for safety */
        for (uint32_t d = 0; d < t->n_dims; d++) {
            if (p + 8 > end) return TN_ERR_INVALID_ARGS;
            t->dims[d] = read_u64(&p);
        }
        if (p + 12 > end) return TN_ERR_INVALID_ARGS;
        t->type   = (GGUFType)read_u32(&p);
        t->offset = read_u64(&p);

        if (looks_like_vision_tensor(t->name)) hdr->is_multimodal = 1;
    }

    /* Align data region to 32 bytes (GGUF v2+ spec) */
    size_t header_end = (size_t)(p - (const uint8_t *)mapped_ptr);
    size_t alignment  = (hdr->version >= 2) ? 32 : 1;
    size_t data_start = (header_end + alignment - 1) & ~(alignment - 1);

    if (data_start > mapped_size) return TN_ERR_INVALID_ARGS;

    hdr->data_region      = (const uint8_t *)mapped_ptr + data_start;
    hdr->data_region_size = mapped_size - data_start;

    /* Resolve tensor data pointers */
    for (uint64_t i = 0; i < hdr->n_tensors; i++) {
        GGUFTensor *t = &hdr->tensors[i];
        size_t sz     = tensor_size_bytes(t);
        /* 2026-10-18: also require the whole tensor to fit. Only the start
         * offset was checked, so a truncated file handed kernels a pointer
         * whose tail ran past the mapping. Q2_0's size is informational
         * (PrismML packing, see gguf_block_size), so it keeps the old rule. */
        int fits = t->offset <= hdr->data_region_size &&
                   (t->type == GGUF_TYPE_Q2_0 || sz <= hdr->data_region_size - t->offset);
        if (!fits) {
            if (t->offset <= hdr->data_region_size)
                TN_LOG_INF("gguf",
                           "[gguf] tensor '%s' extends past the end of the file (truncated?)\n",
                           t->name);
            t->data       = NULL;
            t->size_bytes = 0;
        } else {
            t->data       = (const uint8_t *)hdr->data_region + t->offset;
            t->size_bytes = sz;
        }
    }

    return TN_OK;
}

const GGUFTensor *gguf_find_tensor(const GGUFHeader *hdr, const char *name) {
    for (uint64_t i = 0; i < hdr->n_tensors; i++) {
        if (strcmp(hdr->tensors[i].name, name) == 0) return &hdr->tensors[i];
    }
    return NULL;
}

int gguf_meta_str_copy(const GGUFHeader *hdr, const char *key, char *out, size_t cap) {
    if (!hdr || !key || !out || cap == 0) return -1;
    for (uint64_t i = 0; i < hdr->n_meta; i++) {
        if (strcmp(hdr->meta[i].key, key) == 0 && hdr->meta[i].val_type == GGUF_VAL_STRING) {
            const GGUFMeta *m = &hdr->meta[i];
            size_t n          = m->val.string.len < cap - 1 ? (size_t)m->val.string.len : cap - 1;
            memcpy(out, m->val.string.str, n);
            out[n] = '\0';
            return (int)m->val.string.len;
        }
    }
    return -1;
}

uint32_t gguf_meta_u32(const GGUFHeader *hdr, const char *key, uint32_t default_val) {
    for (uint64_t i = 0; i < hdr->n_meta; i++) {
        if (strcmp(hdr->meta[i].key, key) == 0) {
            const GGUFMeta *m = &hdr->meta[i];
            switch (m->val_type) {
                case GGUF_VAL_UINT32:
                    return m->val.u32;
                case GGUF_VAL_INT32:
                    return (uint32_t)m->val.i32;
                case GGUF_VAL_UINT64:
                    return (uint32_t)m->val.u64;
                default:
                    break;
            }
        }
    }
    return default_val;
}

int32_t gguf_meta_i32(const GGUFHeader *hdr, const char *key, int32_t default_val) {
    for (uint64_t i = 0; i < hdr->n_meta; i++) {
        if (strcmp(hdr->meta[i].key, key) == 0) {
            const GGUFMeta *m = &hdr->meta[i];
            switch (m->val_type) {
                case GGUF_VAL_INT32:
                    return m->val.i32;
                case GGUF_VAL_UINT32:
                    return (int32_t)m->val.u32;
                case GGUF_VAL_INT64:
                    return (int32_t)m->val.i64;
                default:
                    break;
            }
        }
    }
    return default_val;
}

float gguf_meta_f32(const GGUFHeader *hdr, const char *key, float default_val) {
    for (uint64_t i = 0; i < hdr->n_meta; i++) {
        if (strcmp(hdr->meta[i].key, key) == 0) {
            const GGUFMeta *m = &hdr->meta[i];
            switch (m->val_type) {
                case GGUF_VAL_FLOAT32:
                    return m->val.f32;
                case GGUF_VAL_FLOAT64:
                    return (float)m->val.f64;
                case GGUF_VAL_UINT32:
                    return (float)m->val.u32;
                case GGUF_VAL_INT32:
                    return (float)m->val.i32;
                case GGUF_VAL_UINT64:
                    return (float)m->val.u64;
                default:
                    break;
            }
        }
    }
    return default_val;
}

const GGUFMeta *gguf_meta_find(const GGUFHeader *hdr, const char *key) {
    for (uint64_t i = 0; i < hdr->n_meta; i++) {
        if (strcmp(hdr->meta[i].key, key) == 0) return &hdr->meta[i];
    }
    return NULL;
}

uint64_t gguf_meta_array_count(const GGUFHeader *hdr, const char *key) {
    const GGUFMeta *m = gguf_meta_find(hdr, key);
    if (!m || m->val_type != GGUF_VAL_ARRAY) return 0;
    return m->val.array.count;
}

const void *gguf_meta_array_data(const GGUFHeader *hdr, const char *key,
                                 GGUFValType *elem_type_out) {
    const GGUFMeta *m = gguf_meta_find(hdr, key);
    if (!m || m->val_type != GGUF_VAL_ARRAY) return NULL;
    if (elem_type_out) *elem_type_out = m->val.array.elem_type;
    return m->val.array.data;
}

float gguf_type_bpe(GGUFType t) {
    size_t bs = gguf_type_block_bytes(t);
    size_t be = gguf_type_block_elems(t);
    if (be == 0) return 0.0f;
    return (float)bs / (float)be;
}

void gguf_header_free(GGUFHeader *hdr) {
    if (!hdr) return;
    /* Only GGUF_VAL_STRING metadata entries heap-allocate (parse_meta_entry
     * mallocs a NUL-terminated copy since on-disk GGUF strings aren't
     * NUL-terminated); array/tensor data are zero-copy pointers into the
     * mmap and owned by mapped_file_close, not by this struct. */
    for (uint64_t i = 0; i < hdr->n_meta; i++) {
        if (hdr->meta[i].val_type == GGUF_VAL_STRING) {
            free(hdr->meta[i].val.string.str);
        }
    }
}
