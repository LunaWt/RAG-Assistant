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

  it('keeps one block for a batch that reports several results', () => {
    let blocks: Block[] = [];
    blocks = applyEvent(blocks, {
      type: 'tool_start',
      name: ['web_search', 'web_search'],
      args: [{ query: 'a' }, { query: 'b' }],
    });
    blocks = applyEvent(blocks, { type: 'tool_hits', query: 'a', hits: [hit] });
    blocks = applyEvent(blocks, { type: 'tool_hits', query: 'b', hits: [hit] });
    expect(blocks).toHaveLength(1);
    expect(blocks[0]).toEqual({
      type: 'tool',
      names: ['web_search', 'web_search'],
      args: [{ query: 'a' }, { query: 'b' }],
      running: true,
      results: [
        { query: 'a', hits: [hit] },
        { query: 'b', hits: [hit] },
      ],
    });
  });

  it('closes the tool block as soon as the model speaks again', () => {
    let blocks: Block[] = [];
    blocks = applyEvent(blocks, { type: 'tool_start', name: ['web_search'], args: [{}] });
    blocks = applyEvent(blocks, { type: 'tool_hits', query: 'a', hits: [hit] });
    blocks = applyEvent(blocks, { type: 'thought_delta', text: 'got it' });
    expect(blocks).toHaveLength(2);
    expect(blocks[0]).toMatchObject({ type: 'tool', running: false });
  });

  it('closes a tool block that never reported when the answer ends', () => {
    let blocks: Block[] = [];
    blocks = applyEvent(blocks, { type: 'tool_start', name: ['calculator'], args: [{}] });
    blocks = applyEvent(blocks, { type: 'done' });
    expect(blocks[0]).toMatchObject({ type: 'tool', running: false, results: [] });
  });

  it('starts a finished tool block when hits arrive with no tool running', () => {
    const blocks = applyEvent([], { type: 'tool_hits', query: 'x', hits: [hit] });
    expect(blocks).toEqual([
      { type: 'tool', names: [], args: [], running: false, results: [{ query: 'x', hits: [hit] }] },
    ]);
  });

  it('gives every tool round its own block', () => {
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
