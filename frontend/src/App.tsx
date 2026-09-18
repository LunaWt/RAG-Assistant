import * as stylex from '@stylexjs/stylex';
import { useEffect, useRef, useState } from 'react';
import { applyEvent } from './blocks';
import { AssistantMessage, UserMessage } from './Message';
import { streamChat } from './stream';
import type { Block, ChatMessage } from './types';

const styles = stylex.create({
  page: {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    backgroundColor: '#212121',
    color: '#ececec',
  },
  timeline: {
    flex: 1,
    overflowY: 'auto',
    width: '100%',
    maxWidth: '58rem',
    marginInline: 'auto',
    paddingInline: '1rem',
    paddingBlock: '1rem',
    display: 'flex',
    flexDirection: 'column',
    gap: '1rem',
  },
  greeting: { textAlign: 'center', fontSize: '1.75rem', marginBlock: '2.5rem' },
  error: { color: '#f85149' },
  composer: {
    display: 'flex',
    gap: '0.5rem',
    width: '100%',
    maxWidth: '58rem',
    marginInline: 'auto',
    paddingInline: '1rem',
    paddingBlock: '1rem',
  },
  input: {
    flex: 1,
    resize: 'none',
    borderRadius: '1.25rem',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: '#565869',
    backgroundColor: '#2f2f2f',
    color: '#f0f0f0',
    fontSize: '1rem',
    fontFamily: 'inherit',
    paddingInline: '1rem',
    paddingBlock: '0.75rem',
  },
  button: {
    borderRadius: '1.25rem',
    borderStyle: 'none',
    paddingInline: '1.25rem',
    backgroundColor: { default: '#ececec', ':disabled': '#4a4a4a' },
    color: { default: '#212121', ':disabled': '#8b949e' },
    cursor: { default: 'pointer', ':disabled': 'default' },
  },
});

export function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState('');
  const [live, setLive] = useState<Block[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);
  const streaming = live !== null;

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: 'end' });
  }, [messages, live]);

  async function send() {
    const query = draft.trim();
    if (!query || streaming) return;

    const history = messages;
    setMessages([...history, { role: 'user', content: query }]);
    setDraft('');
    setError(null);

    // `blocks` is the source of truth while the stream runs; setLive only mirrors it into
    // the render, because reading React state back inside this loop would see a stale value.
    let blocks: Block[] = [];
    setLive(blocks);
    const controller = new AbortController();
    abort.current = controller;

    try {
      for await (const event of streamChat(query, history, controller.signal)) {
        if (event.type === 'error') {
          setError(event.message);
          return;
        }
        if (event.type === 'done') {
          setMessages((previous) => [...previous, { role: 'assistant', blocks }]);
          return;
        }
        blocks = event.type === 'stream_reset' ? [] : applyEvent(blocks, event);
        setLive(blocks);
      }
      setError('The answer did not finish, so it was not kept. Try again.');
    } catch (failure) {
      if (!controller.signal.aborted) {
        setError(failure instanceof Error ? failure.message : String(failure));
      }
    } finally {
      setLive(null);
      abort.current = null;
    }
  }

  return (
    <div {...stylex.props(styles.page)}>
      <main {...stylex.props(styles.timeline)}>
        {messages.length === 0 && !streaming && (
          <h1 {...stylex.props(styles.greeting)}>RAG Agent</h1>
        )}
        {messages.map((message, index) =>
          message.role === 'user' ? (
            <UserMessage key={index} text={message.content} />
          ) : (
            <AssistantMessage key={index} blocks={message.blocks} />
          ),
        )}
        {live && <AssistantMessage blocks={live} streaming />}
        {error && <p {...stylex.props(styles.error)}>{error}</p>}
        <div ref={bottom} />
      </main>

      <form
        {...stylex.props(styles.composer)}
        onSubmit={(submit) => {
          submit.preventDefault();
          void send();
        }}
      >
        <textarea
          {...stylex.props(styles.input)}
          value={draft}
          rows={1}
          placeholder="Ask anything…"
          onChange={(change) => setDraft(change.target.value)}
          onKeyDown={(key) => {
            if (key.key === 'Enter' && !key.shiftKey) {
              key.preventDefault();
              void send();
            }
          }}
        />
        {streaming ? (
          <button
            type="button"
            {...stylex.props(styles.button)}
            onClick={() => abort.current?.abort()}
          >
            Stop
          </button>
        ) : (
          <button type="submit" {...stylex.props(styles.button)} disabled={!draft.trim()}>
            Send
          </button>
        )}
      </form>
    </div>
  );
}
