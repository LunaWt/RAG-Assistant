import * as stylex from '@stylexjs/stylex';
import Markdown from 'react-markdown';
import type { Block, ToolBlock } from './types';

const TOOL_LABELS: Record<string, string> = {
  search_knowledge_base: 'Searching the knowledge base…',
  web_search: 'Searching the web…',
  calculator: 'Calculating…',
};

const pulse = stylex.keyframes({
  '0%': { opacity: 0.45 },
  '50%': { opacity: 1 },
  '100%': { opacity: 0.45 },
});

const styles = stylex.create({
  bubbleRow: { display: 'flex', justifyContent: 'flex-end' },
  bubble: {
    backgroundColor: '#2f2f2f',
    border: '1px solid #3d3d3d',
    borderRadius: 18,
    padding: '0.5rem 0.95rem',
    maxWidth: '85%',
    whiteSpace: 'pre-wrap',
    overflowWrap: 'break-word',
  },
  assistant: { display: 'flex', flexDirection: 'column', gap: '0.5rem' },
  details: { color: '#c9d1d9' },
  summary: { color: '#8b949e', fontSize: '0.85rem', cursor: 'pointer' },
  status: { color: '#8b949e', fontSize: '0.9rem', animationName: pulse, animationDuration: '1.2s', animationIterationCount: 'infinite' },
  card: {
    border: '1px solid #30363d',
    borderRadius: 12,
    backgroundColor: '#161b22',
    margin: '0.5rem 0',
    overflow: 'hidden',
  },
  cardHeader: { padding: '10px 14px', fontSize: '0.9rem', borderBottom: '1px solid #30363d' },
  hit: {
    display: 'flex',
    justifyContent: 'space-between',
    gap: 12,
    padding: '10px 14px',
    color: '#e6edf3',
    textDecoration: 'none',
    borderTop: '1px solid #21262d',
    backgroundColor: { default: null, ':hover': '#21262d' },
  },
  hitDomain: { color: '#8b949e', fontSize: '0.75rem', flexShrink: 0 },
  toolArgs: { color: '#8b949e', fontSize: '0.8rem', margin: 0 },
});

function domainOf(href: string): string {
  try {
    return new URL(href).hostname.replace(/^www\./, '');
  } catch {
    return href;
  }
}

function ToolCard({ block }: { block: ToolBlock }) {
  return (
    <div>
      {block.names.map((name, i) => (
        <p key={i} {...stylex.props(styles.toolArgs)}>
          {TOOL_LABELS[name] ?? `Running ${name}…`} {JSON.stringify(block.args[i] ?? {})}
        </p>
      ))}
      {block.results.map((result, i) => (
        <div key={i} {...stylex.props(styles.card)}>
          <div {...stylex.props(styles.cardHeader)}>🔍 {result.query}</div>
          {result.hits.map((hit, j) => (
            <a key={j} {...stylex.props(styles.hit)} href={hit.href} target="_blank" rel="noreferrer">
              <span>{hit.title || hit.href}</span>
              <span {...stylex.props(styles.hitDomain)}>{domainOf(hit.href)}</span>
            </a>
          ))}
        </div>
      ))}
    </div>
  );
}

function activityLabel(blocks: Block[]): string {
  for (let i = blocks.length - 1; i >= 0; i--) {
    const block = blocks[i];
    if (block.type === 'tool' && block.running) {
      const name = block.names[0];
      return name ? (TOOL_LABELS[name] ?? `Running ${name}…`) : 'Running tools…';
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

export function AssistantMessage({ blocks, streaming }: { blocks: Block[]; streaming?: boolean }) {
  const answers = blocks.flatMap((b) => (b.type === 'answer' ? [b] : []));
  const reasoning = blocks.filter((b) => b.type !== 'answer');
  return (
    <div {...stylex.props(styles.assistant)}>
      {streaming && answers.length === 0 ? (
        <span {...stylex.props(styles.status)}>{activityLabel(blocks)}</span>
      ) : (
        reasoning.length > 0 && (
          <details {...stylex.props(styles.details)}>
            <summary {...stylex.props(styles.summary)}>Reasoning</summary>
            {reasoning.map((block, i) => {
              if (block.type === 'thought') return <Markdown key={i}>{block.text}</Markdown>;
              if (block.type === 'tool') return <ToolCard key={i} block={block} />;
              return null;
            })}
          </details>
        )
      )}
      {answers.map((block, i) => (
        <Markdown key={i}>{block.text}</Markdown>
      ))}
    </div>
  );
}
