import type { Block, ChatEvent } from './types';

function replacingLast(blocks: Block[], block: Block): Block[] {
  return [...blocks.slice(0, -1), block];
}

// One tool_start can announce several parallel calls, each reporting its own tool_hits, and the
// protocol never says how many are coming. So a tool block stays open until the model speaks
// again, and the same array comes back when there was nothing to close.
function closingTools(blocks: Block[]): Block[] {
  if (!blocks.some((block) => block.type === 'tool' && block.running)) return blocks;
  return blocks.map((block) =>
    block.type === 'tool' && block.running ? { ...block, running: false } : block,
  );
}

export function applyEvent(blocks: Block[], event: ChatEvent): Block[] {
  switch (event.type) {
    case 'thought_delta': {
      const base = closingTools(blocks);
      const last = base.at(-1);
      return last?.type === 'thought'
        ? replacingLast(base, { ...last, text: last.text + event.text })
        : [...base, { type: 'thought', text: event.text }];
    }

    case 'text_delta': {
      const base = closingTools(blocks);
      const last = base.at(-1);
      return last?.type === 'answer'
        ? replacingLast(base, { ...last, text: last.text + event.text })
        : [...base, { type: 'answer', text: event.text }];
    }

    case 'tool_start':
      return [
        ...closingTools(blocks),
        { type: 'tool', names: event.name, args: event.args, running: true, results: [] },
      ];

    case 'tool_hits': {
      const result = { query: event.query, hits: event.hits };
      for (let i = blocks.length - 1; i >= 0; i--) {
        const block = blocks[i];
        if (block.type === 'tool' && block.running) {
          return blocks.with(i, { ...block, results: [...block.results, result] });
        }
      }
      return [
        ...blocks,
        { type: 'tool', names: [], args: [], running: false, results: [result] },
      ];
    }

    case 'done':
      return closingTools(blocks);

    default:
      return blocks;
  }
}
