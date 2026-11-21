<script lang="ts">
  import AsciiLogo from './AsciiLogo.svelte';
  import { tick } from 'svelte';
  import MessageBubble from './MessageBubble.svelte';
  import ImageUpload from './ImageUpload.svelte';
  import { streamChatCompletion, cancelGeneration } from './api';
  import { errorText } from './errors';
  import { requestParams, systemMessage } from './settings';
  import type { ChatMessage, ServerConfig, Settings, StreamSummary } from './types';

  /** A message as shown: what was said, plus the image, queue position and result of its request. */
  interface UiMessage extends ChatMessage {
    imageDataUrl?: string | null;
    queued?: number;
    meta?: StreamSummary & { best_of_n: number };
  }

  export let settings: Settings;
  export let config: ServerConfig;
  export let brand = '';
  export let logoAscii = ''; // block wordmark (GET /v1/config ui.logo_ascii), '' = text only
  export let visionCapable = false;

  let messages: UiMessage[] = [];
  let input = '';
  let imageDataUrl: string | null = null;
  let generating = false;
  let currentRequestId: string | null = null;
  let errorMessage: string | null = null;
  let scrollEl: HTMLElement | undefined;

  async function scrollToBottom(): Promise<void> {
    await tick();
    scrollEl?.scrollTo({ top: scrollEl.scrollHeight, behavior: 'smooth' });
  }

  async function send(): Promise<void> {
    const text = input.trim();
    if (!text || generating) return;

    messages = [...messages, { role: 'user', content: text, imageDataUrl }];
    messages = [...messages, { role: 'assistant', content: '' }];
    const assistantIndex = messages.length - 1;
    input = '';
    const attachedImage = imageDataUrl;
    imageDataUrl = null;
    generating = true;
    errorMessage = null;
    await scrollToBottom();

    const history: ChatMessage[] = messages
      .slice(0, assistantIndex)
      .filter((m) => m.content && typeof m.content === 'string')
      .map((m) => ({ role: m.role, content: m.content }));
    const sys = systemMessage(settings, config);
    if (sys) history.unshift(sys);

    const { id, done } = streamChatCompletion(
      { messages: history, options: requestParams(settings, config), imageDataUrl: attachedImage },
      (delta) => {
        messages[assistantIndex].content += delta;
        messages[assistantIndex].queued = 0;
        messages = messages; // trigger reactivity
        scrollToBottom();
      },
      ({ queued }) => {
        messages[assistantIndex].queued = queued;
        messages = messages;
      }
    );
    id.then((rid) => { currentRequestId = rid; });

    try {
      const summary = await done;
      if (summary && messages[assistantIndex]) {
        const opts = requestParams(settings, config);
        const bestOf = opts.best_of;
        messages[assistantIndex].meta = {
          ...summary,
          best_of_n: typeof bestOf === 'number' ? bestOf : 1,
        };
        messages = messages;
      }
    } catch (err) {
      errorMessage = errorText(err);
    }
    generating = false;
    currentRequestId = null;
  }

  async function stop(): Promise<void> {
    if (currentRequestId) await cancelGeneration(currentRequestId);
  }

  export function clearConversation(): void {
    if (generating) return;
    messages = [];
    errorMessage = null;
  }

  function onKeydown(e: KeyboardEvent): void {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }
</script>

<div class="chat">
  <div class="messages" bind:this={scrollEl}>
    {#if messages.length === 0}
      <div class="empty-state">
        {#if logoAscii}<div class="hero-logo"><AsciiLogo rows={logoAscii} label={brand} size="lg" /></div>{/if}
        <p>Ask {brand} anything.</p>
        <p class="hint">Shift+Enter for a new line · Enter to send</p>
      </div>
    {/if}
    {#each messages as m, i (i)}
      <MessageBubble
        role={m.role}
        content={m.content}
        imageDataUrl={m.imageDataUrl}
        meta={m.meta}
        queued={m.queued ?? 0}
        {brand}
        streaming={generating && i === messages.length - 1 && m.role === 'assistant'}
      />
    {/each}
    {#if errorMessage}
      <div class="error" role="alert"><strong>Error:</strong> {errorMessage}</div>
    {/if}
  </div>

  <div class="composer">
    {#if visionCapable}
      <ImageUpload bind:imageDataUrl />
    {/if}
    <textarea
      placeholder={`Message ${brand}…`}
      bind:value={input}
      on:keydown={onKeydown}
      rows="1"
    ></textarea>
    {#if generating}
      <button class="stop-btn" on:click={stop}>Stop</button>
    {:else}
      <button class="send-btn" on:click={send} disabled={!input.trim()}>Send</button>
    {/if}
  </div>
</div>

<style>
  .chat {
    display: flex;
    flex-direction: column;
    flex: 1;
    min-height: 0;
  }
  .messages {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    padding: 1.25rem;
  }
  .hero-logo {
    display: flex;
    justify-content: center;
    margin-bottom: 1.25rem;
    max-width: 100%;
    overflow: hidden;
  }
  .empty-state {
    margin: auto;
    text-align: center;
    color: var(--text-dim);
  }
  .hint {
    font-size: 0.8rem;
  }
  .error {
    align-self: flex-start;
    color: #e5484d;
    font-size: 0.85rem;
  }
  .composer {
    display: flex;
    align-items: flex-end;
    gap: 0.5rem;
    padding: 1rem 1.25rem;
    border-top: 1px solid var(--border);
    background: var(--panel);
  }
  textarea {
    flex: 1;
    resize: none;
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    font-family: inherit;
    font-size: 0.95rem;
    max-height: 10rem;
  }
  textarea:focus {
    outline: none;
    border-color: var(--accent);
  }
  .send-btn, .stop-btn {
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.1rem;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
  }
  .send-btn {
    background: var(--accent);
    color: var(--accent-contrast);
  }
  .send-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .stop-btn {
    background: transparent;
    border: 1px solid #e5484d;
    color: #e5484d;
  }
</style>
