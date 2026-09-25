import * as stylex from '@stylexjs/stylex';
import Markdown from 'react-markdown';
import { colors } from './tokens.stylex';
import type { Block, Hit, ToolBlock, ToolResult } from './types';

const TOOL_LABELS: Record<string, string> = {
  search_knowledge_base: 'Searching the knowledge base…',
  web_search: 'Searching the web…',
  calculator: 'Calculating…',
};

const TOOL_NAMES: Record<string, string> = {
  search_knowledge_base: 'Knowledge base',
  web_search: 'Web search',
  calculator: 'Calculator',
};

const pulse = stylex.keyframes({
  '0%': { opacity: 0.45 },
  '50%': { opacity: 1 },
  '100%': { opacity: 0.45 },
});

const styles = stylex.create({
  bubbleRow: { display: 'flex', justifyContent: 'flex-end' },
  bubble: {
    backgroundColor: colors.raised,
    borderRadius: 16,
    paddingBlock: '0.6rem',
    paddingInline: '1rem',
    maxWidth: '85%',
    whiteSpace: 'pre-wrap',
    overflowWrap: 'break-word',
  },
  assistant: { display: 'flex', flexDirection: 'column', gap: '0.5rem', lineHeight: 1.6 },
  details: { color: colors.muted },
  summary: { color: colors.muted, fontSize: '0.85rem', cursor: 'pointer' },
  status: {
    color: colors.muted,
    fontSize: '0.9rem',
    animationName: pulse,
    animationDuration: '1.2s',
    animationIterationCount: 'infinite',
  },
  card: {
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: colors.line,
    borderRadius: 12,
    backgroundColor: colors.surface,
    overflow: 'hidden',
  },
  call: {
    borderTopWidth: { default: 1, ':first-child': 0 },
    borderTopStyle: 'solid',
    borderTopColor: colors.line,
  },
  row: {
    display: 'flex',
    alignItems: 'baseline',
    gap: '0.6rem',
    paddingBlock: '0.55rem',
    paddingInline: '0.9rem',
    fontSize: '0.875rem',
  },
  clickable: {
    cursor: 'pointer',
    backgroundColor: { default: null, ':hover': colors.raised },
  },
  mark: { width: '1rem', textAlign: 'center', color: colors.muted, flexShrink: 0 },
  pending: { animationName: pulse, animationDuration: '1.2s', animationIterationCount: 'infinite' },
  failed: { color: colors.danger },
  toolName: { color: colors.text, flexShrink: 0 },
  argument: {
    color: colors.muted,
    flex: 1,
    minWidth: 0,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  faint: { color: colors.muted, fontSize: '0.8rem', flexShrink: 0 },
  errorText: {
    color: colors.danger,
    fontSize: '0.8rem',
    margin: 0,
    paddingInlineStart: '2.5rem',
    paddingInlineEnd: '0.9rem',
    paddingBottom: '0.55rem',
  },
  hit: {
    display: 'flex',
    gap: '0.75rem',
    paddingBlock: '0.5rem',
    paddingInlineStart: '2.5rem',
    paddingInlineEnd: '0.9rem',
    fontSize: '0.85rem',
    borderTopWidth: 1,
    borderTopStyle: 'solid',
    borderTopColor: colors.line,
  },
  link: {
    justifyContent: 'space-between',
    color: colors.text,
    textDecoration: 'none',
    backgroundColor: { default: null, ':hover': colors.raised },
  },
  passage: { flexDirection: 'column', gap: '0.2rem' },
});

function domainOf(href: string): string {
  try {
    return new URL(href).hostname.replace(/^www\./, '');
  } catch {
    return href;
  }
}

function argumentOf(args: Record<string, unknown>): string {
  const main = args.query ?? args.expression;
  if (typeof main !== 'string') return JSON.stringify(args);
  return typeof args.filename === 'string' ? `${main} · ${args.filename}` : main;
}

function outcomeOf(result: ToolResult | null): string {
  if (result === null) return '';
  if (result.status === 'error') return 'failed';
  if (result.status === 'empty') return 'nothing found';
  const count = result.hits.length;
  if (count === 0) return '';
  return count === 1 ? '1 source' : `${count} sources`;
}

function markOf(result: ToolResult | null): string {
  if (result === null) return '•';
  return { ok: '✓', empty: '–', error: '✕' }[result.status];
}

function Hits({ hits }: { hits: Hit[] }) {
  return hits.map((hit, i) =>
    hit.href ? (
      <a
        key={i}
        {...stylex.props(styles.hit, styles.link)}
        href={hit.href}
        target="_blank"
        rel="noreferrer"
      >
        <span>{hit.title || hit.href}</span>
        <span {...stylex.props(styles.faint)}>{domainOf(hit.href)}</span>
      </a>
    ) : (
      <div key={i} {...stylex.props(styles.hit, styles.passage)}>
        <span>{hit.title}</span>
        <span {...stylex.props(styles.faint)}>{hit.snippet}</span>
      </div>
    ),
  );
}

function ToolCall({
  name,
  args,
  result,
}: {
  name: string;
  args: Record<string, unknown>;
  result: ToolResult | null;
}) {
  const row = (
    <>
      <span
        {...stylex.props(
          styles.mark,
          result === null && styles.pending,
          result?.status === 'error' && styles.failed,
        )}
      >
        {markOf(result)}
      </span>
      <span {...stylex.props(styles.toolName)}>{TOOL_NAMES[name] ?? name}</span>
      <span {...stylex.props(styles.argument)}>{argumentOf(args)}</span>
      <span {...stylex.props(styles.faint)}>{outcomeOf(result)}</span>
    </>
  );
  if (result && result.hits.length > 0) {
    return (
      <details {...stylex.props(styles.call)}>
        <summary {...stylex.props(styles.row, styles.clickable)}>{row}</summary>
        <Hits hits={result.hits} />
      </details>
    );
  }
  return (
    <div {...stylex.props(styles.call)}>
      <div {...stylex.props(styles.row)}>{row}</div>
      {result?.status === 'error' && <p {...stylex.props(styles.errorText)}>{result.error}</p>}
    </div>
  );
}

function ToolCard({ block }: { block: ToolBlock }) {
  return (
    <div {...stylex.props(styles.card)}>
      {block.names.map((name, i) => (
        <ToolCall key={i} name={name} args={block.args[i]} result={block.results[i]} />
      ))}
    </div>
  );
}

function activityLabel(blocks: Block[]): string {
  for (let i = blocks.length - 1; i >= 0; i--) {
    const block = blocks[i];
    if (block.type !== 'tool') continue;
    const pending = block.results.indexOf(null);
    if (pending !== -1) {
      const name = block.names[pending];
      return TOOL_LABELS[name] ?? `Running ${name}…`;
    }
  }
  return 'Thinking…';
}

export function UserMessage({ text }: { text: string }) {
  return (
    <div {...stylex.props(styles.bubbleRow)}>
      <div {...stylex.props(styles.bubble)}>{text}</div>
    </div>
  );
}

// Thoughts stay folded under Reasoning. Tool cards and answer text keep their arrival order
// outside it, so a status and its sources are visible without opening anything.
export function AssistantMessage({ blocks, streaming }: { blocks: Block[]; streaming?: boolean }) {
  const thoughts = blocks.flatMap((b) => (b.type === 'thought' ? [b] : []));
  const working = streaming && !blocks.some((b) => b.type === 'answer');
  return (
    <div {...stylex.props(styles.assistant)}>
      {!working && thoughts.length > 0 && (
        <details {...stylex.props(styles.details)}>
          <summary {...stylex.props(styles.summary)}>Reasoning</summary>
          {thoughts.map((block, i) => (
            <div key={i} className="prose">
              <Markdown>{block.text}</Markdown>
            </div>
          ))}
        </details>
      )}
      {blocks.map((block, i) => {
        if (block.type === 'tool') return <ToolCard key={i} block={block} />;
        if (block.type === 'answer') {
          return (
            <div key={i} className="prose">
              <Markdown>{block.text}</Markdown>
            </div>
          );
        }
        return null;
      })}
      {working && <span {...stylex.props(styles.status)}>{activityLabel(blocks)}</span>}
    </div>
  );
}
