#ifndef TN_PATCH_EXTRACT_H
#define TN_PATCH_EXTRACT_H

/**
 * Extracts patches from an image.
 *
 * @param image       The input image as a flat float array (img_size * img_size * 3).
 * @param patches     The output buffer for patches. Must be pre-allocated.
 *                    Size: (img_size/patch_size)^2 * (patch_size * patch_size * 3) floats.
 * @param img_size    The width (and height) of the square image.
 * @param patch_size  The size of each square patch (e.g., 14 or 16).
 * @param num_patches An output pointer that receives the total number of patches extracted.
 */
void extract_patches(const float *image, float *patches, int img_size, int patch_size, int *num_patches);

#endif /* TN_PATCH_EXTRACT_H */
