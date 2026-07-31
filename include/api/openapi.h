#ifndef TN_OPENAPI_H
#define TN_OPENAPI_H

/*
 * Phase 22 — Static OpenAPI 3.0 description + a hand-rolled GET /docs page.
 * Deliberately not a vendored Swagger-UI bundle, so this route has zero
 * dependency on the webui/ npm pipeline (Phase 22.2).
 */

/* Static OpenAPI 3.0 JSON document describing every route. Never NULL. */
const char *openapi_json(void);

/* Static HTML+JS docs page that fetches openapi_json() from /openapi.json
 * and renders a simple route list. Never NULL. */
const char *openapi_docs_html(void);

#endif /* TN_OPENAPI_H */
