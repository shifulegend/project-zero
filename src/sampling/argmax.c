#include "sampling/sampling.h"

int sample_argmax(const float *logits, int vocab_size) {
  int best_idx = 0;
  float best_val = logits[0];
  for (int i = 1; i < vocab_size; i++) {
    if (logits[i] > best_val) {
      best_val = logits[i];
      best_idx = i;
    }
  }
  return best_idx;
}
