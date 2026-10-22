#ifndef TN_SHA256_H
#define TN_SHA256_H

/*
 * core/sha256 — FIPS 180-4 SHA-256 (2026-10-22), for verifying downloads
 * (the setup wizard's starter models) without an external tool.
 */

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t h[8];
    uint64_t len;          /* bytes hashed so far */
    unsigned char buf[64];
    size_t   nbuf;
} Sha256;

void sha256_init(Sha256 *s);
void sha256_update(Sha256 *s, const void *data, size_t n);
void sha256_final(Sha256 *s, unsigned char out[32]);

/* Lower-case hex digest of a whole file into hex[65]. Returns 0, or -1 if the
 * file cannot be read. */
int sha256_file_hex(const char *path, char hex[65]);

/* Lower-case hex of 32 digest bytes into hex[65]. */
void sha256_to_hex(const unsigned char d[32], char hex[65]);

#endif /* TN_SHA256_H */
