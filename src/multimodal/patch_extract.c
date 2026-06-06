#include "multimodal/patch_extract.h"

void extract_patches(const float *image, float *patches, int img_size, int patch_size, int *num_patches) {
    int patches_per_row = img_size / patch_size;
    int patches_per_col = img_size / patch_size;
    *num_patches = patches_per_row * patches_per_col;

    int patch_idx = 0;

    for (int py = 0; py < patches_per_col; py++) {
        for (int px = 0; px < patches_per_row; px++) {

            // For each patch, copy the pixels
            int patch_offset = patch_idx * (patch_size * patch_size * 3);
            int pixel_idx = 0;

            for (int y = 0; y < patch_size; y++) {
                for (int x = 0; x < patch_size; x++) {
                    int img_y = py * patch_size + y;
                    int img_x = px * patch_size + x;
                    int img_offset = (img_y * img_size + img_x) * 3;

                    patches[patch_offset + pixel_idx * 3 + 0] = image[img_offset + 0];
                    patches[patch_offset + pixel_idx * 3 + 1] = image[img_offset + 1];
                    patches[patch_offset + pixel_idx * 3 + 2] = image[img_offset + 2];

                    pixel_idx++;
                }
            }
            patch_idx++;
        }
    }
}
