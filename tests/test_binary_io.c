/* tn_rd_*: alignment-safe 32-bit reads at arbitrary byte offsets. */
#include <stdio.h>
#include <string.h>

#include "core/binary_io.h"

static int g_fail;
#define CHECK(c, msg)                                                                              \
    do {                                                                                           \
        if (!(c)) {                                                                                \
            printf("FAIL: %s\n", msg);                                                             \
            g_fail++;                                                                              \
        } else                                                                                     \
            printf("ok: %s\n", msg);                                                               \
    } while (0)

int main(void) {
    unsigned char buf[16] = {0};
    uint32_t u            = 0xA1B2C3D4u;
    int32_t i             = -123456;
    float f               = 3.5f;
    memcpy(buf + 1, &u, 4); /* deliberately misaligned offsets */
    memcpy(buf + 5, &i, 4);
    memcpy(buf + 9, &f, 4);
    CHECK(tn_rd_u32(buf, 1) == 0xA1B2C3D4u, "u32 at an odd offset");
    CHECK(tn_rd_i32(buf, 5) == -123456, "i32 negative value at an odd offset");
    CHECK(tn_rd_f32(buf, 9) == 3.5f, "f32 at an odd offset");
    CHECK(tn_rd_u32(buf + 4, 1) == (uint32_t)(i), "base pointer plus offset");
    printf(g_fail ? "binary_io: %d FAILED\n" : "binary_io: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
