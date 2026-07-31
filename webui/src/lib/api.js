// Phase 22.2 — API client for project-zero's HTTP server.
//
// Uses fetch() + a manual SSE line parser rather than EventSource, because
// EventSource cannot send a POST body and /v1/chat/completions requires one
// (the messages array, sampling params, etc).

/** @returns {Promise<{id: string, object: string, owned_by: string}[]>} */
export async function fetchModels() {
  const res = await fetch('/v1/models');
  if (!res.ok) throw new Error(`GET /v1/models failed: ${res.status}`);
  const json = await res.json();
  return json.data ?? [];
}

/**
 * Streams a chat completion, invoking callbacks as SSE events arrive.
 *
 * @param {object} params
 * @param {{role: string, content: string}[]} params.messages
 * @param {number} params.temperature
 * @param {number} params.top_p
 * @param {number} params.max_tokens
 * @param {string=} params.imageDataUrl - base64 data: URL, if an image is attached
 * @param {(delta: string) => void} onToken
 * @returns {{ id: Promise<string|null>, done: Promise<void>, abort: () => void }}
 */
export function streamChatCompletion({ messages, temperature, top_p, max_tokens, imageDataUrl }, onToken) {
  const controller = new AbortController();
  let resolveId;
  const idPromise = new Promise((resolve) => { resolveId = resolve; });

  let requestMessages = messages;
  if (imageDataUrl) {
    // OpenAI "content parts" form — the last user message gets the image
    // attached alongside its text.
    const last = messages[messages.length - 1];
    requestMessages = [
      ...messages.slice(0, -1),
      {
        role: last.role,
        content: [
          { type: 'text', text: last.content },
          { type: 'image_url', image_url: { url: imageDataUrl } },
        ],
      },
    ];
  }

  const body = JSON.stringify({
    messages: requestMessages,
    temperature,
    top_p,
    max_tokens,
    stream: true,
  });

  const done = (async () => {
    let response;
    try {
      response = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
        signal: controller.signal,
      });
    } catch (err) {
      resolveId(null);
      if (err.name === 'AbortError') return;
      throw err;
    }

    if (!response.ok || !response.body) {
      resolveId(null);
      const text = await response.text().catch(() => '');
      throw new Error(`Chat completion failed: ${response.status} ${text}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let idResolved = false;

    try {
      for (;;) {
        const { value, done: streamDone } = await reader.read();
        if (streamDone) break;
        buffer += decoder.decode(value, { stream: true });

        let sepIndex;
        while ((sepIndex = buffer.indexOf('\n\n')) !== -1) {
          const rawEvent = buffer.slice(0, sepIndex);
          buffer = buffer.slice(sepIndex + 2);
          const line = rawEvent.trim();
          if (!line.startsWith('data:')) continue;
          const payload = line.slice(5).trim();
          if (payload === '[DONE]') continue;

          let evt;
          try { evt = JSON.parse(payload); } catch { continue; }

          if (!idResolved && evt.id) { idResolved = true; resolveId(evt.id); }
          const delta = evt.choices?.[0]?.delta?.content;
          if (delta) onToken(delta);
        }
      }
    } finally {
      if (!idResolved) resolveId(null);
    }
  })();

  return { id: idPromise, done, abort: () => controller.abort() };
}

/** Cancels an in-flight generation by its "chatcmpl-..." id. */
export async function cancelGeneration(id) {
  if (!id) return;
  await fetch('/v1/chat/completions/cancel', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id }),
  }).catch(() => {});
}
