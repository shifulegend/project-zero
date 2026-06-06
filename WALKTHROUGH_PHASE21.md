# Walkthrough: Phase 21 — OpenAI-Compatible API Layer

## Overview

Phase 21 turns the Project Zero inference engine into a **headless HTTP daemon** that
any OpenAI-compatible frontend can connect to.  After loading a model, the engine can
now run in server mode, accepting chat completion requests and streaming token-by-token
responses via Server-Sent Events (SSE).

This makes the engine compatible with tools like:
- **Cline** (VS Code AI coding assistant)
- **OpenHands** (software engineering agent)
- **SillyTavern** (roleplay frontend)
- **OpenWebUI** (self-hosted ChatGPT alternative)
- **LangChain / LlamaIndex** (RAG frameworks)
- Any HTTP client that speaks the OpenAI chat completions schema

---

## What Was Built

### New source files

| File | Purpose |
|------|---------|
| `src/api/http_server.c` | Minimal POSIX HTTP/1.1 server using raw sockets; routes requests to handlers |
| `src/api/json_parse.c` | Purpose-built JSON parser for the OpenAI chat completions request schema |
| `src/api/chat_compile.c` | Multi-format chat template compiler (ChatML, Llama-3, DeepSeek, Mistral, Raw) |
| `src/api/sse_stream.c` | Server-Sent Events helpers — format and write OpenAI streaming tokens |

### New headers

| File | Purpose |
|------|---------|
| `include/api/api_server.h` | Public API: `api_context_init()`, `api_server_start()`, `api_server_stop()` |
| `include/api/chat_request.h` | `ChatRequest`/`ChatMessage` structures + `chat_request_parse()` |
| `include/api/chat_compile.h` | `ChatTemplateType` enum + `chat_compile_prompt()` |
| `include/api/sse_stream.h` | `sse_write_token()`, `sse_write_done()`, `sse_write_full_response()` |

### Modified files

| File | Change |
|------|--------|
| `src/transformer/generate.c` | Refactored: `generate_with_callback()` is the core loop; `generate()` is a thin stdout wrapper |
| `include/transformer/generate.h` | Added `TokenCallback` typedef and `generate_with_callback()` declaration |
| `include/core/error.h` | Added `TN_ERR_SOCKET_CREATE`, `TN_ERR_SOCKET_BIND`, `TN_ERR_SOCKET_LISTEN`, `TN_ERR_JSON_PARSE` |
| `include/cli/args.h` | Added `server_mode` and `server_port` fields to `CliArgs` |
| `src/cli/args.c` | Parses `--server` and `--port <N>` flags; updated help text |
| `src/cli/main.c` | Starts API server when `--server` flag is set; blocks on `pause()` |
| `Makefile` | Added `build/api/http_server.o` rule with `-D_GNU_SOURCE`; added `build/tests/test_api_server` rule using g++ linker |

### New tests

| File | What it tests |
|------|--------------|
| `tests/test_api_server.c` | JSON parser (basic, multi-turn, escapes, unknown keys, malformed), template detection, ChatML/Llama-3 template compilation, SSE token/done/escaping, tiny-buffer safety |

---

## Architecture: How It Works

```
User starts engine with --server --port 8080
                |
         api_server_start(8080, &ctx)
                |
         Spawns listener_thread_fn()      (background POSIX thread)
                |
         accept() loop on 127.0.0.1:8080
                |
         handle_connection(fd, ctx)       (per connection, sequential)
                |
         http_parse()                     (method, path, headers, body)
                |
         Route dispatch
           ├─→ GET  /v1/models            → JSON model list
           ├─→ GET  /health               → {"status":"ok"}
           └─→ POST /v1/chat/completions  → handle_chat_completions()
                                              |
                                         chat_request_parse()   (JSON → ChatRequest)
                                              |
                                         chat_compile_prompt()  (messages → prompt string)
                                              (skipped for GGUF models with embedded template)
                                              |
                                         generate_with_callback()  (token-by-token inference)
                                              |
                                         streaming_token_callback()
                                              ├─ stream=true  → sse_write_token() per token
                                              └─ stream=false → accumulate → sse_write_full_response()
                                              |
                                         sse_write_done()       (streaming: emit [DONE])
```

---

## API Endpoints

### `GET /v1/models`
```json
{
  "object": "list",
  "data": [{"id": "local-adaptive-engine", "object": "model", "owned_by": "project-zero"}]
}
```

### `GET /health`
```json
{"status": "ok"}
```

### `POST /v1/chat/completions`

**Request body:**
```json
{
  "model": "local",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is 2+2?"}
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 256,
  "stream": true
}
```

**Streaming response** (`stream: true`):
```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"4"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

**Non-streaming response** (`stream: false`):
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "4."}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}
```

---

## Chat Template Logic

The template selection follows this priority order:

1. **GGUF embedded template** (preferred): If the tokenizer carries a Jinja2 `chat_template`
   string (loaded from the GGUF file), `generate_with_callback()` applies it internally via
   `chat_template_apply()`. The API layer passes only the last user message content.

2. **Heuristic detection** (fallback): For legacy `.bin` tokenizers without an embedded
   template, `chat_template_detect(model_name)` selects a format from the `model` field
   of the request. Supported formats:

   | Model name contains | Template used |
   |---------------------|--------------|
   | `llama-3` / `llama3` | Llama-3 `<\|start_header_id\|>` |
   | `mistral` / `mixtral` | Mistral `[INST]...[/INST]` |
   | `deepseek` | DeepSeek `User: / Assistant:` |
   | anything else | ChatML `<\|im_start\|>` (default) |

---

## Threading and Safety

- The HTTP listener runs on a **single background thread** (sequential connections).
- The inference engine (`RunState`, `TransformerWeights`) is **not thread-safe**.
  Concurrent requests would require per-request RunState cloning — deferred to Phase 23
  (Continuous Batching). For now, one request is processed at a time.
- The server binds to **127.0.0.1 only** (loopback). It is not reachable from other
  machines on the network by default.

---

## Security Properties

| Property | Implementation |
|----------|----------------|
| Loopback-only | `htonl(INADDR_LOOPBACK)` — `127.0.0.1` |
| Request size cap | `HTTP_MAX_BODY_BYTES = 512 KiB` |
| JSON string limits | `CHAT_MAX_CONTENT = 64 KiB` per message |
| No shell injection | Inference runs in-process, no `exec()` |
| JSON output safety | All token text JSON-escaped before writing to wire |

---

## How to Use

### Start the API server

```bash
./adaptive_ai_engine \
  --model   models/bitnet-b1.58-2B-4T.bin \
  --tokenizer models/bitnet-b1.58-2B-4T_tokenizer_proper.bin \
  --server \
  --port 8080
```

### Test with curl

```bash
# Health check
curl http://127.0.0.1:8080/health

# List models
curl http://127.0.0.1:8080/v1/models

# Non-streaming completion
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role":"user","content":"What is the capital of France?"}],
    "max_tokens": 64,
    "stream": false
  }'

# Streaming completion
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role":"user","content":"Count to 5."}],
    "stream": true
  }'
```

### Configure Cline / OpenHands / OpenWebUI

Point the "OpenAI API Base URL" to `http://127.0.0.1:8080` and set the model to `local`.
No API key required.

---

## Running the Tests

```bash
make build/tests/test_api_server
./build/tests/test_api_server
```

Expected output:
```
=== Phase 21 API Server Tests ===
[PASS] chat_request_init
[PASS] json_parse_basic
[PASS] json_parse_multiturn
[PASS] json_parse_escapes
[PASS] json_parse_unknown_keys
[PASS] json_parse_malformed
[PASS] chat_template_detect
[PASS] compile_chatml
[PASS] compile_llama3
[PASS] sse_write_token
[PASS] sse_write_done
[PASS] sse_json_escaping
[PASS] compile_tiny_buffer_no_crash

=== All API tests passed ===
```

---

## Key Design Decisions

### No external dependencies
Phase 21 uses only POSIX system calls (`socket`, `bind`, `listen`, `accept`, `read`, `write`,
`pthread_create`) already present in the engine's POSIX baseline. No mongoose, no libmicrohttpd,
no cJSON library is required.

### Callback-based generation
Rather than forking or piping the output of `generate()`, the token output path was refactored
into `generate_with_callback()`. This avoids any buffering overhead and delivers each token
to the HTTP connection immediately, achieving true first-token latency as low as the prefill
time.

### Sequential connections
A single listener thread processes connections one at a time. This is intentional:
- The inference `RunState` is not thread-safe (no locking).
- Adding parallelism requires per-request RunState cloning (Phase 23 work).
- For single-user local deployment, sequential is perfectly adequate.

---

## What's Next (Phase 22+)

| Phase | Feature | Builds on |
|-------|---------|-----------|
| Phase 22 | SSM/Mamba architecture router | Phase 21 server + Phase 6 forward pass |
| Phase 23 | Continuous batching + per-request RunState | Phase 21 API layer |
| Phase 24 | Dynamic context scaling (YaRN/NTK) | Phase 21 |
| Phase 26 | Radix tree prompt caching (zero-compute repeat calls) | Phase 21 server |
