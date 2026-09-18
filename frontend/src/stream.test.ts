import { describe, expect, it } from 'vitest';
import { ndjsonEvents } from './stream';
import type { ChatEvent } from './types';

function bodyOf(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
}

async function collect(chunks: string[]): Promise<ChatEvent[]> {
  const events: ChatEvent[] = [];
  for await (const event of ndjsonEvents(bodyOf(chunks))) events.push(event);
  return events;
}

describe('ndjsonEvents', () => {
  it('reassembles an event split across two network chunks', async () => {
    const events = await collect(['{"type":"text_de', 'lta","text":"hi"}\n']);
    expect(events).toEqual([{ type: 'text_delta', text: 'hi' }]);
  });

  it('yields several events that arrived in one chunk', async () => {
    const events = await collect(['{"type":"done"}\n{"type":"done"}\n']);
    expect(events).toHaveLength(2);
  });

  it('yields a last line that has no trailing newline', async () => {
    const events = await collect(['{"type":"done"}']);
    expect(events).toEqual([{ type: 'done' }]);
  });

  it('skips blank lines', async () => {
    const events = await collect(['\n{"type":"done"}\n\n']);
    expect(events).toEqual([{ type: 'done' }]);
  });

  it('keeps multi-byte characters whole when a chunk splits them', async () => {
    const whole = new TextEncoder().encode('{"type":"text_delta","text":"привет"}\n');
    const encoder = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(whole.slice(0, 30));
        controller.enqueue(whole.slice(30));
        controller.close();
      },
    });
    const events: ChatEvent[] = [];
    for await (const event of ndjsonEvents(encoder)) events.push(event);
    expect(events).toEqual([{ type: 'text_delta', text: 'привет' }]);
  });
});
