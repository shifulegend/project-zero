#include "transformer/gen_params.h"
#include <string.h>

void gen_params_default(GenerationParams *gp) {
    if (!gp) return;
    memset(gp, 0, sizeof(*gp));
    gp->max_tokens = 256;
    sampler_settings_default(&gp->sampler);
    gp->seed = -1;
    watermark_config_default(&gp->watermark);
}

const char *gen_finish_reason_str(GenFinishReason r) {
    switch (r) {
        case GEN_FINISH_LENGTH:    return "length";
        case GEN_FINISH_CANCELLED: return "cancelled";
        case GEN_FINISH_ERROR:     return "error";
        case GEN_FINISH_STOP:
        default:                   return "stop";
    }
}
