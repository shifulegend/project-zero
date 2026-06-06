#ifndef TN_IMAGE_LOAD_H
#define TN_IMAGE_LOAD_H

#include "core/error.h"

// Loads an image from path, converts to float RGB [0,1], and resizes to target_res x target_res
// Returns TN_OK on success, or TN_ERR_IMAGE_LOAD on failure.
// The caller is responsible for freeing *pixels (using standard free() since it's float,
// though if we need SIMD we might use tn_aligned_free later, but let's stick to standard free
// or we can use our aligned allocator).
TernaryError load_image(const char *path, float **pixels, int *width, int *height, int target_res);

#endif /* TN_IMAGE_LOAD_H */
