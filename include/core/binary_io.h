#ifndef TN_CORE_BINARY_IO_H
#define TN_CORE_BINARY_IO_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Read a 32-bit value at byte offset `off` of a mapped/in-memory buffer. memcpy keeps the
 * access alignment-safe; the value is in the host's byte order (little-endian on every
 * supported target), as the file formats read with these helpers are. The caller checks
 * that off + 4 is inside the buffer. */
static inline uint32_t tn_rd_u32(const void *base, size_t off) {
    uint32_t v;
    memcpy(&v, (const char *)base + off, sizeof v);
    return v;
}

static inline int32_t tn_rd_i32(const void *base, size_t off) {
    int32_t v;
    memcpy(&v, (const char *)base + off, sizeof v);
    return v;
}

static inline float tn_rd_f32(const void *base, size_t off) {
    float v;
    memcpy(&v, (const char *)base + off, sizeof v);
    return v;
}

#endif
