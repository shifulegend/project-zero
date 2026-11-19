/* Without any tn_simd_init() call the dispatch pointers must already work (portable scalar code),
 * and give the same results as the scalar functions they point at. */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "math/elementwise.h"
#include "math/rmsnorm.h"
#include "math/simd_dispatch.h"
#include "math/softmax.h"

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
    CHECK(tn_vec_dot && tn_rmsnorm && tn_softmax && tn_vec_add && tn_vec_mul && tn_vec_scale &&
              tn_silu && tn_relu2 && tn_vec_saxpy && tn_unpack_block && tn_ternary_matmul &&
              tn_ternary_matmul_packed && tn_kvq4_dot && tn_kvq4_axpy && tn_cls_i8_rows &&
              tn_cls_i4_rows,
          "every dispatch pointer is non-NULL before tn_simd_init()");
    CHECK(strcmp(tn_simd_backend_name(), "Scalar") == 0,
          "the default backend is reported as Scalar");
    CHECK(tn_quantize_act_i8 == NULL && tn_ternary_matmul_preq == NULL &&
              tn_simd_ternary_preq() == 0,
          "the int8-activation pointers stay NULL and tn_simd_ternary_preq() says so");

    float a[8] = {1, 2, 3, 4, 5, 6, 7, 8}, b[8] = {8, 7, 6, 5, 4, 3, 2, 1}, o[8], r[8];
    CHECK(fabsf(tn_vec_dot(a, b, 8) - vec_dot(a, b, 8)) < 1e-6f, "tn_vec_dot matches vec_dot");
    tn_vec_add(o, a, b, 8);
    vec_add(r, a, b, 8);
    CHECK(memcmp(o, r, sizeof o) == 0, "tn_vec_add matches vec_add");
    float x[8], y[8], w[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    memcpy(x, a, sizeof x);
    memcpy(y, a, sizeof y);
    tn_softmax(x, 8);
    softmax(y, 8);
    CHECK(memcmp(x, y, sizeof x) == 0, "tn_softmax matches softmax");
    tn_rmsnorm(o, a, w, 8, 1e-5f);
    rmsnorm(r, a, w, 8, 1e-5f);
    CHECK(memcmp(o, r, sizeof o) == 0, "tn_rmsnorm matches rmsnorm");
    printf(g_fail ? "simd_default: %d FAILED\n" : "simd_default: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
