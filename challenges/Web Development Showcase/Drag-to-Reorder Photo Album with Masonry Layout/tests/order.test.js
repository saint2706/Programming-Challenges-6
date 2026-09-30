import { describe, expect, it } from 'vitest';
import {
  deserialize,
  isSameOrder,
  moveById,
  moveItem,
  nudge,
  placeRelativeTo,
  reconcile,
  serialize,
} from '../src/order.js';

const abcde = ['a', 'b', 'c', 'd', 'e'];

describe('moveItem', () => {
  it('moves forward and backward', () => {
    expect(moveItem(abcde, 0, 2)).toEqual(['b', 'c', 'a', 'd', 'e']);
    expect(moveItem(abcde, 4, 1)).toEqual(['a', 'e', 'b', 'c', 'd']);
  });
  it('is a no-op when from === to', () => {
    expect(moveItem(abcde, 2, 2)).toEqual(abcde);
  });
  it('clamps the target index', () => {
    expect(moveItem(abcde, 1, 99)).toEqual(['a', 'c', 'd', 'e', 'b']);
    expect(moveItem(abcde, 3, -5)).toEqual(['d', 'a', 'b', 'c', 'e']);
  });
  it('ignores out-of-range sources and empty lists', () => {
    expect(moveItem(abcde, 9, 0)).toEqual(abcde);
    expect(moveItem(abcde, -1, 0)).toEqual(abcde);
    expect(moveItem([], 0, 0)).toEqual([]);
  });
  it('never mutates its input', () => {
    const input = [...abcde];
    const out = moveItem(input, 0, 4);
    expect(input).toEqual(abcde);
    expect(out).not.toBe(input);
  });
});

describe('moveById / nudge', () => {
  it('moves by id and ignores unknown ids', () => {
    expect(moveById(abcde, 'c', 0)).toEqual(['c', 'a', 'b', 'd', 'e']);
    expect(moveById(abcde, 'zzz', 0)).toEqual(abcde);
  });
  it('nudges one step and stops at the ends', () => {
    expect(nudge(abcde, 'c', -1)).toEqual(['a', 'c', 'b', 'd', 'e']);
    expect(nudge(abcde, 'c', 1)).toEqual(['a', 'b', 'd', 'c', 'e']);
    expect(nudge(abcde, 'a', -1)).toEqual(abcde);
    expect(nudge(abcde, 'e', 1)).toEqual(abcde);
    expect(nudge(abcde, 'zzz', 1)).toEqual(abcde);
  });
});

describe('placeRelativeTo', () => {
  it('places before a later target', () => {
    expect(placeRelativeTo(abcde, 'a', 'd', 'left')).toEqual(['b', 'c', 'a', 'd', 'e']);
  });
  it('places after a later target', () => {
    expect(placeRelativeTo(abcde, 'a', 'd', 'right')).toEqual(['b', 'c', 'd', 'a', 'e']);
  });
  it('places before / after an earlier target', () => {
    expect(placeRelativeTo(abcde, 'e', 'b', 'left')).toEqual(['a', 'e', 'b', 'c', 'd']);
    expect(placeRelativeTo(abcde, 'e', 'b', 'right')).toEqual(['a', 'b', 'e', 'c', 'd']);
  });
  it("dropping on the neighbour's facing edge is a no-op, not an off-by-one", () => {
    expect(placeRelativeTo(abcde, 'b', 'a', 'right')).toEqual(abcde);
    expect(placeRelativeTo(abcde, 'b', 'c', 'left')).toEqual(abcde);
  });
  it('supports before/after edge names', () => {
    expect(placeRelativeTo(abcde, 'a', 'c', 'before')).toEqual(['b', 'a', 'c', 'd', 'e']);
    expect(placeRelativeTo(abcde, 'a', 'c', 'after')).toEqual(['b', 'c', 'a', 'd', 'e']);
  });
  it('reaches both ends of the list', () => {
    expect(placeRelativeTo(abcde, 'c', 'a', 'left')).toEqual(['c', 'a', 'b', 'd', 'e']);
    expect(placeRelativeTo(abcde, 'c', 'e', 'right')).toEqual(['a', 'b', 'd', 'e', 'c']);
  });
  it('ignores self-drops and unknown ids', () => {
    expect(placeRelativeTo(abcde, 'a', 'a', 'left')).toEqual(abcde);
    expect(placeRelativeTo(abcde, 'x', 'a', 'left')).toEqual(abcde);
    expect(placeRelativeTo(abcde, 'a', 'x', 'left')).toEqual(abcde);
  });
  it('always yields a permutation of the input', () => {
    for (const drag of abcde) {
      for (const target of abcde) {
        for (const edge of ['left', 'right']) {
          expect([...placeRelativeTo(abcde, drag, target, edge)].sort()).toEqual(abcde);
        }
      }
    }
  });
});

describe('reconcile', () => {
  it('keeps a valid saved order', () => {
    expect(reconcile(['c', 'a', 'b'], ['a', 'b', 'c'])).toEqual(['c', 'a', 'b']);
  });
  it('drops removed photos', () => {
    expect(reconcile(['c', 'gone', 'a', 'b'], ['a', 'b', 'c'])).toEqual(['c', 'a', 'b']);
  });
  it('appends new photos in canonical order', () => {
    expect(reconcile(['b', 'a'], ['a', 'b', 'c', 'd'])).toEqual(['b', 'a', 'c', 'd']);
  });
  it('drops duplicates', () => {
    expect(reconcile(['a', 'a', 'b', 'a'], ['a', 'b'])).toEqual(['a', 'b']);
  });
  it('falls back to canonical for unusable input', () => {
    for (const bad of [null, undefined, 'abc', 42, {}, [1, 2, null]]) {
      expect(reconcile(bad, ['a', 'b'])).toEqual(['a', 'b']);
    }
  });
  it('handles an empty canonical list', () => {
    expect(reconcile(['a'], [])).toEqual([]);
  });
});

describe('serialize / deserialize', () => {
  it('round-trips', () => {
    expect(deserialize(serialize(abcde))).toEqual(abcde);
  });
  it('rejects non-strings, garbage and wrong shapes', () => {
    for (const bad of [null, undefined, 5, '', '{nope', 'null', '[]', '{"v":1}', '{"v":1,"order":"abc"}']) {
      expect(deserialize(bad)).toBeNull();
    }
  });
  it('rejects a different schema version', () => {
    expect(deserialize('{"v":99,"order":["a"]}')).toBeNull();
  });
  it('filters non-string entries', () => {
    expect(deserialize('{"v":1,"order":["a",1,null,"b"]}')).toEqual(['a', 'b']);
  });
});

describe('isSameOrder', () => {
  it('compares element-wise', () => {
    expect(isSameOrder(['a', 'b'], ['a', 'b'])).toBe(true);
    expect(isSameOrder(['a', 'b'], ['b', 'a'])).toBe(false);
    expect(isSameOrder(['a'], ['a', 'b'])).toBe(false);
  });
});
