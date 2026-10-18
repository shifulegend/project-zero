#include "sampling/logprob.h"
#include <math.h>

double logprob_logsumexp(const float *logits, int n) {
    if (!logits || n <= 0) return -INFINITY;
    float mx = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    if (!isfinite(mx)) return (double)mx;
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += exp((double)(logits[i] - mx));
    return (double)mx + log(sum);
}
