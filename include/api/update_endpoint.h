#ifndef TN_UPDATE_ENDPOINT_H
#define TN_UPDATE_ENDPOINT_H

#include "update/update_check.h"

/*
 * update_endpoint.h — Phase 38: JSON for GET /v1/update and
 * POST /v1/update/check. Reads the update service (update/update_check.h);
 * never touches the generation mutex. GET also kicks a background check when
 * the cached answer is stale, so a long-running server stays current without
 * a timer thread.
 */

/* `force` = 1 for POST /v1/update/check (rate-limited by the service). `svc` may be NULL. */
char *api_update_json(UpdateService *svc, int force);

#endif /* TN_UPDATE_ENDPOINT_H */
