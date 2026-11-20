#ifndef TN_PZ_CONFIG_SCHEMA_H
#define TN_PZ_CONFIG_SCHEMA_H

#include "config/pz_config.h"

/* Upper bound on the number of schema keys, for arrays indexed by key. */
#define PZ_CFG_SCHEMA_MAX 64

extern const PzCfgKey g_pz_cfg_schema[];
extern const int g_pz_cfg_schema_len;

#endif
