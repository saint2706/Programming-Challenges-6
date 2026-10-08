import { describe, expect, it } from 'vitest';
import { moveRelative } from '../src/lib/dnd.js';

const order = ['a', 'b', 'c', 'd'];

describe('moveRelative', () => {
  it('drops an item above a target', () => {
    expect(moveRelative(order, 'd', 'b', 'top')).toEqual(['a', 'd', 'b', 'c']);
  });

  it('drops an item below a target', () => {
    expect(moveRelative(order, 'a', 'c', 'bottom')).toEqual(['b', 'c', 'a', 'd']);
  });

  it('works at both ends', () => {
    expect(moveRelative(order, 'c', 'a', 'top')).toEqual(['c', 'a', 'b', 'd']);
    expect(moveRelative(order, 'a', 'd', 'bottom')).toEqual(['b', 'c', 'd', 'a']);
  });

  it('returns the same order when nothing would change', () => {
    expect(moveRelative(order, 'b', 'b', 'top')).toBe(order);
    expect(moveRelative(order, 'b', 'a', 'bottom')).toBe(order);
    expect(moveRelative(order, 'a', 'b', 'top')).toBe(order);
  });

  it('ignores unknown ids and never mutates its input', () => {
    expect(moveRelative(order, 'x', 'a', 'top')).toBe(order);
    expect(moveRelative(order, 'a', 'x', 'top')).toBe(order);
    expect(order).toEqual(['a', 'b', 'c', 'd']);
  });

  it('always keeps every item exactly once', () => {
    for (const id of order) {
      for (const target of order) {
        for (const edge of ['top', 'bottom']) {
          expect([...moveRelative(order, id, target, edge)].sort()).toEqual(['a', 'b', 'c', 'd']);
        }
      }
    }
  });
});
