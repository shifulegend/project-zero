#ifndef TN_AUTH_H
#define TN_AUTH_H

/*
 * Phase 22 — Optional API-key auth ("Authorization: Bearer <key>").
 * Off by default (AuthConfig.api_key == NULL) so existing loopback
 * workflows are unaffected; enabled via --api-key.
 */

#include "api/server_config.h"

/* Returns 1 if the request is authorized (auth disabled, or
 * authorization_header_value is "Bearer <api_key>"), 0 otherwise.
 * authorization_header_value may be NULL (header absent). */
int auth_check_request(const AuthConfig *cfg, const char *authorization_header_value);

#endif /* TN_AUTH_H */
