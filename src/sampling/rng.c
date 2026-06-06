#include "sampling/rng.h"

void rng_seed(unsigned long long *state, unsigned long long seed) {
  /* Ensure non-zero state */
  *state = (seed == 0) ? 1 : seed;
}

float rng_float(unsigned long long *state) {
  /* Xorshift64* — simple, fast, and sufficient for sampling */
  unsigned long long s = *state;
  s ^= s >> 12;
  s ^= s << 25;
  s ^= s >> 27;
  *state = s;
  /* Map to [0, 1) by taking upper bits and dividing */
  return (float)((s * 0x2545F4914F6CDD1DULL) >> 33) / (float)(1u << 31);
}
