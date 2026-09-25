import * as stylex from '@stylexjs/stylex';
import { useEffect, useRef, useState } from 'react';
import { applyEvent } from './blocks';
import { AssistantMessage, UserMessage } from './Message';
import { streamChat } from './stream';
import { colors } from './tokens.stylex';
import type { Block, ChatMessage } from './types';

/** A question that got no answer, kept so Retry can send it again with the same history. */
type Unanswered = { query: string; history: ChatMessage[]; error: string | null };

const styles = stylex.create({
  page: {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    backgroundColor: colors.page,
    color: colors.text,
  },
  // The whole width scrolls, so the scrollbar sits at the window edge, not beside the column.
  scroller: {
    flex: 1,
    overflowY: 'auto',
    '::-webkit-scrollbar': { width: 8 },
    '::-webkit-scrollbar-thumb': { backgroundColor: colors.lineStrong, borderRadius: 4 },
  },
  timeline: {
    boxSizing: 'border-box',
    width: '100%',
    maxWidth: '64rem',
    marginInline: 'auto',
    paddingInline: '1rem',
    paddingBlock: '1.5rem',
    display: 'flex',
    flexDirection: 'column',
    gap: '1.25rem',
  },
  greeting: {
    textAlign: 'center',
    fontFamily: 'ui-serif, Georgia, Cambria, "Times New Roman", serif',
    fontWeight: 400,
    fontSize: '2.25rem',
    marginBlock: '3rem',
  },
  notice: { display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.9rem' },
  error: { color: colors.danger },
  stopped: { color: colors.muted },
  retry: {
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: colors.lineStrong,
    borderRadius: 8,
    paddingBlock: '0.3rem',
    paddingInline: '0.8rem',
    backgroundColor: { default: 'transparent', ':hover': colors.raised },
    color: colors.text,
    fontSize: '0.85rem',
    cursor: 'pointer',
  },
  composerWrap: {
    width: '100%',
    maxWidth: '64rem',
    marginInline: 'auto',
    paddingInline: '1rem',
    paddingBottom: '1.25rem',
    boxSizing: 'border-box',
  },
  composer: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: colors.lineStrong,
    borderRadius: 18,
    backgroundColor: colors.surface,
    boxShadow: '0 1px 0 #141414',
    paddingBlock: '0.75rem',
    paddingInline: '0.9rem',
  },
  input: {
    resize: 'none',
    borderStyle: 'none',
    outline: 'none',
    backgroundColor: 'transparent',
    color: colors.text,
    fontSize: '1rem',
    fontFamily: 'inherit',
    lineHeight: 1.5,
    paddingInline: '0.25rem',
  },
  actions: { display: 'flex', justifyContent: 'flex-end' },
  button: {
    borderRadius: 8,
    borderStyle: 'none',
    paddingBlock: '0.4rem',
    paddingInline: '0.9rem',
    fontSize: '0.9rem',
    fontWeight: 500,
    cursor: { default: 'pointer', ':disabled': 'default' },
  },
  send: {
    backgroundColor: { default: colors.accent, ':disabled': colors.raised },
    color: { default: colors.page, ':disabled': colors.muted },
  },
  stop: { backgroundColor: colors.raised, color: colors.text },
});

export function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState('');
  const [live, setLive] = useState<Block[] | null>(null);
  const [unanswered, setUnanswered] = useState<Unanswered | null>(null);
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);
  const streaming = live !== null;

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: 'end' });
  }, [messages, live, unanswered]);

  async function ask(query: string, history: ChatMessage[]) {
    setUnanswered(null);
    const fail = (error: string | null) => setUnanswered({ query, history, error });

    // `blocks` is the source of truth while the stream runs; setLive only mirrors it into
    // the render, because reading React state back inside this loop would see a stale value.
    let blocks: Block[] = [];
    setLive(blocks);
    const controller = new AbortController();
    abort.current = controller;

    try {
      for await (const event of streamChat(query, history, controller.signal)) {
        if (event.type === 'error') {
          fail(event.message);
          return;
        }
        if (event.type === 'done') {
          const finished = blocks;
          setMessages((previous) => [...previous, { role: 'assistant', blocks: finished }]);
          return;
        }
        blocks = event.type === 'stream_reset' ? [] : applyEvent(blocks, event);
        setLive(blocks);
      }
      fail('The answer did not finish, so it was not kept.');
    } catch (failure) {
      if (controller.signal.aborted) fail(null);
      else fail(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setLive(null);
      abort.current = null;
    }
  }

  function send() {
    const query = draft.trim();
    if (!query || streaming) return;
    setMessages([...messages, { role: 'user', content: query }]);
    setDraft('');
    void ask(query, messages);
  }

  return (
    <div {...stylex.props(styles.page)}>
      <div {...stylex.props(styles.scroller)}>
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
          {unanswered && (
            <div {...stylex.props(styles.notice)}>
              <span {...stylex.props(unanswered.error ? styles.error : styles.stopped)}>
                {unanswered.error ?? 'Stopped.'}
              </span>
              <button
                type="button"
                {...stylex.props(styles.retry)}
                onClick={() => void ask(unanswered.query, unanswered.history)}
              >
                Retry
              </button>
            </div>
          )}
          <div ref={bottom} />
        </main>
      </div>

      <div {...stylex.props(styles.composerWrap)}>
        <form
          {...stylex.props(styles.composer)}
          onSubmit={(submit) => {
            submit.preventDefault();
            send();
          }}
        >
          <textarea
            {...stylex.props(styles.input)}
            value={draft}
            rows={2}
            placeholder="Ask anything…"
            onChange={(change) => setDraft(change.target.value)}
            onKeyDown={(key) => {
              // isComposing: an IME is mid-word, so Enter is picking a candidate, not sending.
              if (key.key === 'Enter' && !key.shiftKey && !key.nativeEvent.isComposing) {
                key.preventDefault();
                send();
              }
            }}
          />
          <div {...stylex.props(styles.actions)}>
            {streaming ? (
              <button
                type="button"
                {...stylex.props(styles.button, styles.stop)}
                onClick={() => abort.current?.abort()}
              >
                Stop
              </button>
            ) : (
              <button
                type="submit"
                {...stylex.props(styles.button, styles.send)}
                disabled={!draft.trim()}
              >
                Send
              </button>
            )}
          </div>
        </form>
      </div>
    </div>
  );
}
