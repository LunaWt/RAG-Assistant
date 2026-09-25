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
      return [
        ...blocks,
        { type: 'tool', names: event.name, args: event.args, results: event.name.map(() => null) },
      ];

    // The round a result belongs to is always the last block: the model cannot speak again
    // until every call of the round has reported.
    case 'tool_result':
      return last?.type === 'tool'
        ? replacingLast(blocks, { ...last, results: last.results.with(event.index, event.result) })
        : blocks;

    default:
      return blocks;
  }
}
