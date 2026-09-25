import { describe, expect, it } from 'vitest';
import { applyEvent } from './blocks';
import type { Block, ToolResult } from './types';

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

  it('puts each result at its call index, whatever order the calls finish in', () => {
    let blocks: Block[] = [];
    blocks = applyEvent(blocks, {
      type: 'tool_start',
      name: ['web_search', 'calculator'],
      args: [{ query: 'a' }, { expression: '1 +' }],
    });
    const failed: ToolResult = { status: 'error', hits: [], error: 'Syntax error in expression' };
    blocks = applyEvent(blocks, { type: 'tool_result', index: 1, result: failed });
    expect(blocks[0]).toMatchObject({ results: [null, failed] });
    blocks = applyEvent(blocks, {
      type: 'tool_result',
      index: 0,
      result: { status: 'ok', hits: [hit] },
    });
    expect(blocks).toEqual([
      {
        type: 'tool',
        names: ['web_search', 'calculator'],
        args: [{ query: 'a' }, { expression: '1 +' }],
        results: [{ status: 'ok', hits: [hit] }, failed],
      },
    ]);
  });

  it('gives every tool round its own block', () => {
    let blocks: Block[] = [];
    blocks = applyEvent(blocks, { type: 'tool_start', name: ['web_search'], args: [{}] });
    blocks = applyEvent(blocks, {
      type: 'tool_result',
      index: 0,
      result: { status: 'ok', hits: [hit] },
    });
    blocks = applyEvent(blocks, { type: 'tool_start', name: ['calculator'], args: [{}] });
    expect(blocks).toHaveLength(2);
  });

  it('leaves the timeline alone for control events', () => {
    const before: Block[] = [{ type: 'answer', text: 'Hi' }];
    expect(applyEvent(before, { type: 'done' })).toBe(before);
  });
});
