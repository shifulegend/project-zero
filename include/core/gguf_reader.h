#ifndef TN_GGUF_READER_H
#define TN_GGUF_READER_H

/*
 * gguf_reader.h — Phase 34.2: GGUF Format Reader
 *
 * Parses the GGUF binary format used by llama.cpp and the broader
 * open-source LLM ecosystem. Supports GGUF v1, v2, and v3.
 *
 * Scope: header parsing + tensor metadata only. Weight data is accessed
 * via the returned tensor descriptors (zero-copy pointer into the mmap).
 * Architecture-specific weight loading is handled by the caller.
 */

#include "core/error.h"
#include <stddef.h>
#include <stdint.h>

/* ── GGUF constants ─────────────────────────────────────────────────────── */
#define GGUF_MAGIC         0x46554747u  /* "GGUF" LE */
#define GGUF_MAX_TENSORS   4096
#define GGUF_MAX_KEY_LEN   256
#define GGUF_MAX_META_KEYS 512

/* GGUF quantization types (matches ggml_type enum) */
typedef enum {
    GGUF_TYPE_F32    = 0,
    GGUF_TYPE_F16    = 1,
    GGUF_TYPE_Q4_0   = 2,
    GGUF_TYPE_Q4_1   = 3,
    GGUF_TYPE_Q5_0   = 6,
    GGUF_TYPE_Q5_1   = 7,
    GGUF_TYPE_Q8_0   = 8,
    GGUF_TYPE_Q8_1   = 9,
    GGUF_TYPE_Q2_K   = 10,
    GGUF_TYPE_Q3_K   = 11,
    GGUF_TYPE_Q4_K   = 12,
    GGUF_TYPE_Q5_K   = 13,
    GGUF_TYPE_Q6_K   = 14,
    GGUF_TYPE_Q8_K   = 15,
    GGUF_TYPE_IQ2_XXS = 16,
    GGUF_TYPE_IQ4_NL  = 20,
    GGUF_TYPE_I8     = 24,
    GGUF_TYPE_I16    = 25,
    GGUF_TYPE_I32    = 26,
    GGUF_TYPE_I64    = 27,
    GGUF_TYPE_F64    = 28,
    GGUF_TYPE_BF16   = 30,
    GGUF_TYPE_Q2_0   = 42, /* legacy 2-bit block quant (ggml block_q2_0, block=64) —
                               used by PrismML's Qwen3.6-based "Bonsai" ternary GGUF releases */
    GGUF_TYPE_COUNT
} GGUFType;

/* GGUF metadata value types */
typedef enum {
    GGUF_VAL_UINT8   = 0,
    GGUF_VAL_INT8    = 1,
    GGUF_VAL_UINT16  = 2,
    GGUF_VAL_INT16   = 3,
    GGUF_VAL_UINT32  = 4,
    GGUF_VAL_INT32   = 5,
    GGUF_VAL_FLOAT32 = 6,
    GGUF_VAL_BOOL    = 7,
    GGUF_VAL_STRING  = 8,
    GGUF_VAL_ARRAY   = 9,
    GGUF_VAL_UINT64  = 10,
    GGUF_VAL_INT64   = 11,
    GGUF_VAL_FLOAT64 = 12,
} GGUFValType;

/* ── Tensor descriptor (zero-copy into mmap) ────────────────────────────── */
typedef struct {
    char     name[GGUF_MAX_KEY_LEN];
    uint32_t n_dims;
    uint64_t dims[4];          /* up to 4 dimensions, dims[0] = innermost */
    GGUFType type;
    uint64_t offset;           /* byte offset from data region start */
    const void *data;          /* pointer into mmap — valid while file mapped */
    size_t   size_bytes;       /* total bytes including padding */
} GGUFTensor;

/* ── Metadata key-value entry ───────────────────────────────────────────── */
typedef struct {
    char        key[GGUF_MAX_KEY_LEN];
    GGUFValType val_type;
    union {
        uint8_t  u8;
        int8_t   i8;
        uint16_t u16;
        int16_t  i16;
        uint32_t u32;
        int32_t  i32;
        float    f32;
        uint64_t u64;
        int64_t  i64;
        double   f64;
        uint8_t  boolean;
        struct { char *str; uint64_t len; } string; /* heap-allocated, NUL-terminated */
        /* Array: zero-copy pointer into mmap pointing at first element.
         * For string arrays, elements are variable-length [len:u64][bytes].
         * For numeric arrays, elements are packed scalars (direct cast). */
        struct {
            GGUFValType elem_type;
            uint64_t    count;
            const void *data;   /* points into mmap — valid while file mapped */
        } array;
    } val;
} GGUFMeta;

/* ── Parsed GGUF header ─────────────────────────────────────────────────── */
typedef struct {
    uint32_t    version;           /* GGUF version (1, 2, or 3) */
    uint64_t    n_tensors;
    uint64_t    n_meta;

    GGUFMeta    meta[GGUF_MAX_META_KEYS];
    GGUFTensor  tensors[GGUF_MAX_TENSORS];

    const void *data_region;       /* pointer to start of tensor data */
    size_t      data_region_size;

    /* Convenience fields extracted from metadata */
    char   arch[64];               /* llama.arch value, e.g. "llama", "phi2" */
    int    is_multimodal;          /* 1 if vision tower tensors detected */
} GGUFHeader;

/* ── Public API ─────────────────────────────────────────────────────────── */

/**
 * Parse a GGUF file from a memory-mapped buffer.
 * hdr is populated in-place. O(n_tensors + n_meta).
 */
TernaryError gguf_read_header(GGUFHeader *hdr, const void *mapped_ptr, size_t mapped_size);

/**
 * Frees the heap-allocated NUL-terminated copies made for GGUF_VAL_STRING
 * metadata entries (parse_meta_entry allocates these since on-disk GGUF
 * strings aren't NUL-terminated). Array/tensor data are zero-copy pointers
 * into the caller's mmap and are not touched here — callers still need
 * mapped_file_close (or equivalent) for those. Safe to call on a
 * zero-initialized or already-freed GGUFHeader (n_meta == 0 is a no-op).
 * Must be called once gguf_read_header succeeds, alongside the caller's
 * other cleanup (see src/cli/main.c) — previously omitted, which leaked
 * every string-typed metadata value for the life of the process.
 */
void gguf_header_free(GGUFHeader *hdr);

/**
 * Find a tensor by name. Returns NULL if not found.
 */
const GGUFTensor *gguf_find_tensor(const GGUFHeader *hdr, const char *name);

/**
 * Look up a metadata string value by key. Returns NULL if not found or wrong type.
 */
const char *gguf_meta_str(const GGUFHeader *hdr, const char *key);

/**
 * Look up a metadata uint32 value by key. Returns default_val if not found.
 */
uint32_t gguf_meta_u32(const GGUFHeader *hdr, const char *key, uint32_t default_val);

/**
 * Look up a metadata int32 value by key. Returns default_val if not found.
 */
int32_t gguf_meta_i32(const GGUFHeader *hdr, const char *key, int32_t default_val);

/**
 * Look up a metadata float32 value by key. Returns default_val if not found.
 * Also accepts FLOAT64 and casts to float.
 */
float gguf_meta_f32(const GGUFHeader *hdr, const char *key, float default_val);

/**
 * Find a metadata entry by key. Returns NULL if not found.
 */
const GGUFMeta *gguf_meta_find(const GGUFHeader *hdr, const char *key);

/**
 * Return element count for an array metadata key.
 * Returns 0 if key not found or not an array.
 */
uint64_t gguf_meta_array_count(const GGUFHeader *hdr, const char *key);

/**
 * Return pointer to first element of an array metadata key (zero-copy into mmap).
 * For string arrays, elements are variable-length [len:u64][bytes].
 * For numeric arrays, elements are packed scalars (direct cast).
 * Optionally writes element type to *elem_type_out.
 * Returns NULL if key not found or not an array.
 */
const void *gguf_meta_array_data(const GGUFHeader *hdr, const char *key,
                                  GGUFValType *elem_type_out);

/**
 * Return human-readable name for a GGUF quantization type.
 */
const char *gguf_type_name(GGUFType t);

/**
 * Return bytes-per-element for a GGUF type (approximate for block-quantized types).
 */
float gguf_type_bpe(GGUFType t);

#endif /* TN_GGUF_READER_H */
