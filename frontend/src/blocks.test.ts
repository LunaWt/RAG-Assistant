import { describe, expect, it } from 'vitest';
import { applyEvent } from './blocks';
import type { Block } from './types';

const hit = { title: 'Docs', href: 'https://example.com/docs' };

describe('applyEvent', () => {
  it('merges consecutive deltas of the same kind into one block', () => {
    let blocks: Block[] = [];
    blocks = applyEvent(blocks, { type: 'text_delta', text: 'Hel' });
    blocks = applyEvent(blocks, { type: 'text_delta', text: 'lo' });
    expect(blocks).toEqual([{ type: 'answer', text: 'Hello' }]);
  });

  it('keeps thought, tool and answer as separate blocks in arrival order', () => {
    let blocks: Block[] = [];
    blocks = applyEvent(blocks, { type: 'thought_delta', text: 'searching' });
    blocks = applyEvent(blocks, { type: 'tool_start', name: ['web_search'], args: [{ query: 'x' }] });
    blocks = applyEvent(blocks, { type: 'text_delta', text: 'answer' });
    expect(blocks.map((block) => block.type)).toEqual(['thought', 'tool', 'answer']);
  });

  it('never mutates the array it was given, so React sees a new reference', () => {
    const before: Block[] = [{ type: 'answer', text: 'Hel' }];
    const after = applyEvent(before, { type: 'text_delta', text: 'lo' });
    expect(before).toEqual([{ type: 'answer', text: 'Hel' }]);
    expect(after).not.toBe(before);
    expect(after[0]).not.toBe(before[0]);
  });

  it('attaches hits to the running tool block and stops its spinner', () => {
    let blocks: Block[] = [];
    blocks = applyEvent(blocks, { type: 'tool_start', name: ['web_search'], args: [{ query: 'x' }] });
    blocks = applyEvent(blocks, { type: 'tool_hits', query: 'x', hits: [hit] });
    expect(blocks).toEqual([
      {
        type: 'tool',
        names: ['web_search'],
        args: [{ query: 'x' }],
        running: false,
        results: [{ query: 'x', hits: [hit] }],
      },
    ]);
  });

  it('starts a finished tool block when hits arrive with no tool running', () => {
    const blocks = applyEvent([], { type: 'tool_hits', query: 'x', hits: [hit] });
    expect(blocks).toEqual([
      { type: 'tool', names: [], args: [], running: false, results: [{ query: 'x', hits: [hit] }] },
    ]);
  });

  it('does not reopen a finished tool block: a later tool_start is its own block', () => {
    let blocks: Block[] = [];
    blocks = applyEvent(blocks, { type: 'tool_start', name: ['web_search'], args: [{}] });
    blocks = applyEvent(blocks, { type: 'tool_hits', query: 'x', hits: [hit] });
    blocks = applyEvent(blocks, { type: 'tool_start', name: ['calculator'], args: [{}] });
    expect(blocks).toHaveLength(2);
  });

  it('leaves the timeline alone for control events', () => {
    const before: Block[] = [{ type: 'answer', text: 'Hi' }];
    expect(applyEvent(before, { type: 'done' })).toBe(before);
  });
});
