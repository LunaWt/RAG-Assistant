import type { ChatEvent, ChatMessage } from './types';

/** One JSON object per line, reassembled across whatever chunk boundaries the network picks. */
export async function* ndjsonEvents(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<ChatEvent> {
  const reader = body.getReader();
  // stream: true holds back the tail of a multi-byte character until its next bytes arrive.
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut = buffer.indexOf('\n');
      while (cut !== -1) {
        const line = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 1);
        if (line.trim()) yield JSON.parse(line) as ChatEvent;
        cut = buffer.indexOf('\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim()) yield JSON.parse(buffer) as ChatEvent;
  } finally {
    await reader.cancel();
  }
}

export async function* streamChat(
  query: string,
  history: ChatMessage[],
  signal: AbortSignal,
): AsyncGenerator<ChatEvent> {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, history, session_id: null }),
    signal,
  });
  if (!response.ok) {
    throw new Error(`POST /api/chat answered ${response.status}`);
  }
  if (!response.body) {
    throw new Error('POST /api/chat answered without a body');
  }
  yield* ndjsonEvents(response.body);
}
