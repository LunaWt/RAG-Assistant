import type { Block, ChatEvent } from './types';

function replacingLast(blocks: Block[], block: Block): Block[] {
  return [...blocks.slice(0, -1), block];
}

export function applyEvent(blocks: Block[], event: ChatEvent): Block[] {
  const last = blocks.at(-1);
  switch (event.type) {
    case 'thought_delta':
      return last?.type === 'thought'
        ? replacingLast(blocks, { ...last, text: last.text + event.text })
        : [...blocks, { type: 'thought', text: event.text }];

    case 'text_delta':
      return last?.type === 'answer'
        ? replacingLast(blocks, { ...last, text: last.text + event.text })
        : [...blocks, { type: 'answer', text: event.text }];

    case 'tool_start':
      return last?.type === 'tool' && last.running
        ? replacingLast(blocks, {
            ...last,
            names: [...last.names, ...event.name],
            args: [...last.args, ...event.args],
          })
        : [
            ...blocks,
            {
              type: 'tool',
              names: event.name,
              args: event.args,
              running: true,
              results: [],
            },
          ];

    case 'tool_hits': {
      const result = { query: event.query, hits: event.hits };
      for (let i = blocks.length - 1; i >= 0; i--) {
        const block = blocks[i];
        if (block.type === 'tool' && block.running) {
          return blocks.with(i, {
            ...block,
            running: false,
            results: [...block.results, result],
          });
        }
      }
      return [
        ...blocks,
        { type: 'tool', names: [], args: [], running: false, results: [result] },
      ];
    }

    default:
      return blocks;
  }
}
