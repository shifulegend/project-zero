#include "multimodal/image_load.h"
#include <stdlib.h>
#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

/* Vendored third-party header — suppress its own internal warnings rather
 * than patching upstream code (avoids merge friction on future updates). */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"
#pragma GCC diagnostic pop

#include "memory/aligned_alloc.h"

/* SigLIP normalization: [0,1] → [-1,1]  (mean=0.5, std=0.5)
 * Matches Qwen2-VL / PaliGemma / SigLIP preprocessing convention.
 * n_pixels = width * height (each pixel has 3 channels). */
static void normalize_siglip(float *pixels, int n_pixels) {
    int n = n_pixels * 3;
    for (int i = 0; i < n; i++)
        pixels[i] = pixels[i] * 2.0f - 1.0f;
}

TernaryError load_image(const char *path, float **pixels, int *width, int *height, int target_res) {
    if (!path || !pixels || !width || !height) return TN_ERR_INVALID_ARGS;

    int img_w, img_h, channels;
    // Load as float RGB (3 channels)
    float *img_f = stbi_loadf(path, &img_w, &img_h, &channels, 3);
    if (!img_f) {
        fprintf(stderr, "Failed to load image: %s\n", stbi_failure_reason());
        return TN_ERR_IMAGE_LOAD;
    }

    *width = target_res;
    *height = target_res;

    // Allocate output array. target_res * target_res * 3 floats.
    size_t out_elements;
    if (tn_size_mul3(target_res, target_res, 3, &out_elements)) {
        stbi_image_free(img_f);
        return TN_ERR_OOM;
    }

    // We use aligned alloc to ensure it's safe for SIMD.
    float *out_f = (float *)tn_aligned_alloc(out_elements * sizeof(float), 64);
    if (!out_f) {
        stbi_image_free(img_f);
        return TN_ERR_OOM;
    }

    // Resize using bilinear interpolation
    stbir_resize_float_linear(img_f, img_w, img_h, 0, out_f, target_res, target_res, 0, (stbir_pixel_layout)3);

    stbi_image_free(img_f);

    // Normalize to [-1, 1] using SigLIP convention (matches Qwen2-VL / PaliGemma)
    normalize_siglip(out_f, target_res * target_res);

    *pixels = out_f;
    return TN_OK;
}
