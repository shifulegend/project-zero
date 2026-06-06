#ifndef TN_RNG_H
#define TN_RNG_H

/**
 * Xorshift128+ pseudo-random number generator.
 * Fast, no external dependencies, good enough for sampling.
 */

/**
 * Seed the RNG state. Must be called before rng_float().
 * @param state  Pointer to 64-bit RNG state
 * @param seed   Seed value (must be non-zero)
 */
void rng_seed(unsigned long long *state, unsigned long long seed);

/**
 * Generate a uniform random float in [0, 1).
 * @param state  Pointer to 64-bit RNG state (modified in-place)
 * @return       Random float in [0, 1)
 */
float rng_float(unsigned long long *state);

#endif /* TN_RNG_H */
