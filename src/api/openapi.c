/*
 * Phase 22 — src/api/openapi.c
 * See include/api/openapi.h for responsibility.
 */

#include "api/openapi.h"

static const char OPENAPI_JSON[] =
"{"
  "\"openapi\":\"3.0.0\","
  "\"info\":{\"title\":\"project-zero API\",\"version\":\"1.0.0\","
    "\"description\":\"OpenAI-compatible local inference API.\"},"
  "\"paths\":{"
    "\"/v1/models\":{\"get\":{\"summary\":\"List available models\","
      "\"responses\":{\"200\":{\"description\":\"OK\"}}}},"
    "\"/v1/chat/completions\":{\"post\":{\"summary\":\"Create a chat completion\","
      "\"description\":\"Supports streaming (SSE) and non-streaming modes.\","
      "\"responses\":{\"200\":{\"description\":\"OK\"},\"400\":{\"description\":\"Invalid request\"},"
        "\"401\":{\"description\":\"Unauthorized\"},\"429\":{\"description\":\"Another generation is in flight\"}}}},"
    "\"/v1/chat/completions/cancel\":{\"post\":{\"summary\":\"Cancel an in-flight generation\","
      "\"responses\":{\"200\":{\"description\":\"OK\"},\"404\":{\"description\":\"No matching in-flight request\"}}}},"
    "\"/health\":{\"get\":{\"summary\":\"Liveness check\","
      "\"responses\":{\"200\":{\"description\":\"OK\"}}}},"
    "\"/metrics\":{\"get\":{\"summary\":\"Prometheus text-exposition metrics\","
      "\"responses\":{\"200\":{\"description\":\"OK\"},\"404\":{\"description\":\"Disabled (no --metrics)\"}}}},"
    "\"/docs\":{\"get\":{\"summary\":\"This documentation page\","
      "\"responses\":{\"200\":{\"description\":\"OK\"}}}},"
    "\"/\":{\"get\":{\"summary\":\"Web chat UI\","
      "\"responses\":{\"200\":{\"description\":\"OK\"}}}}"
  "}"
"}";

static const char OPENAPI_DOCS_HTML[] =
"<!doctype html><html><head><meta charset=\"utf-8\">"
"<title>project-zero API docs</title>"
"<style>"
"body{font-family:ui-monospace,Menlo,Consolas,monospace;max-width:760px;margin:2rem auto;padding:0 1rem;"
"background:#0b0d12;color:#e6e8ee}"
"h1{font-size:1.3rem}"
"code{color:#8fd0ff}"
".route{border:1px solid #2a2f3a;border-radius:8px;padding:.75rem 1rem;margin:.6rem 0}"
".method{display:inline-block;font-weight:bold;padding:.1rem .5rem;border-radius:4px;margin-right:.5rem}"
".get{background:#173a2e;color:#7ee2a8}.post{background:#2e2417;color:#f0b869}"
"</style></head><body>"
"<h1>project-zero API</h1>"
"<p>OpenAPI description: <a href=\"/openapi.json\" style=\"color:#8fd0ff\">/openapi.json</a></p>"
"<div id=\"routes\"></div>"
"<script>"
"fetch('/openapi.json').then(r=>r.json()).then(spec=>{"
"const el=document.getElementById('routes');"
"for(const path in spec.paths){"
"for(const method in spec.paths[path]){"
"const op=spec.paths[path][method];"
"const div=document.createElement('div');div.className='route';"
"div.innerHTML='<span class=\"method '+method+'\">'+method.toUpperCase()+'</span>'"
"+'<code>'+path+'</code><div>'+(op.summary||'')+'</div>';"
"el.appendChild(div);"
"}}"
"}).catch(()=>{document.getElementById('routes').textContent='Failed to load /openapi.json';});"
"</script>"
"</body></html>";

const char *openapi_json(void) { return OPENAPI_JSON; }
const char *openapi_docs_html(void) { return OPENAPI_DOCS_HTML; }
