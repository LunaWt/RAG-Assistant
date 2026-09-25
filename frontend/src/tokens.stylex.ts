import * as stylex from '@stylexjs/stylex';

// The claude.ai dark palette, sampled 22 Sep 2026 (~/.claude/frontend-design.md): warm
// near-black layers separated by a border one step lighter, and one accent used once.
export const colors = stylex.defineConsts({
  page: '#151515',
  surface: '#20201f',
  raised: '#2c2c2b',
  line: '#3a3a38',
  lineStrong: '#4d4d4c',
  text: '#f0efec',
  muted: '#8f8f8d',
  accent: '#d97757',
  danger: '#ee7b73',
});
