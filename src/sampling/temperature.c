#include "sampling/sampling.h"
#include "core/platform.h"

#if TN_HAS_AVX512
#include <immintrin.h>
#endif

void apply_temperature(float *logits, int vocab_size, float temperature) {
  /* Guard against near-zero temperature causing inf/NaN */
  if (temperature < 1e-6f) temperature = 1e-6f;
#if TN_HAS_AVX512
  float inv_temp = 1.0f / temperature;
  __m512 t = _mm512_set1_ps(inv_temp);
  int i = 0;
  for (; i + 15 < vocab_size; i += 16) {
    __m512 v = _mm512_loadu_ps(&logits[i]);
    _mm512_storeu_ps(&logits[i], _mm512_mul_ps(v, t));
  }
  for (; i < vocab_size; i++)
    logits[i] *= inv_temp;
#else
  for (int i = 0; i < vocab_size; i++)
    logits[i] /= temperature;
#endif
}
