#ifndef TN_DATA_URL_H
#define TN_DATA_URL_H

/*
 * Phase 22.2 — decodes a "data:<mime>;base64,<payload>" URL (the form the
 * web UI sends for image uploads) into raw bytes.
 */

#include "core/error.h"
#include <stddef.h>

/* Decodes data_url into a heap buffer. Returns TN_OK and sets *out_data
 * (caller must free()) and *out_len on success. Returns
 * TN_ERR_INVALID_ARGS if data_url isn't a "data:...;base64,..." URL, or
 * TN_ERR_OOM on allocation failure. */
TernaryError data_url_decode(const char *data_url, unsigned char **out_data, size_t *out_len);

#endif /* TN_DATA_URL_H */
