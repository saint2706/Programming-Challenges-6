import { describe, it, expect } from 'vitest';
import { STEP_COUNT, clampStep, formatHash, furthestReachable, parseHash, progressPercent } from '../src/lib/navigation.js';

describe('parseHash / formatHash', () => {
  it('round-trips one-based hashes to zero-based indices', () => {
    expect(parseHash('#/step/1')).toBe(0);
    expect(parseHash('#/step/4')).toBe(3);
    expect(formatHash(2)).toBe('#/step/3');
  });
  it('rejects junk', () => {
    for (const h of ['', '#', '#/step/0', '#/step/-1', '#/step/x', '#/step/2/extra', '#step/2', null, undefined]) {
      expect(parseHash(h)).toBeNull();
    }
  });
});

describe('furthestReachable / clampStep', () => {
  it('with nothing verified only step 0 is reachable', () => {
    expect(furthestReachable([])).toBe(0);
    expect(clampStep(4, [])).toBe(0);
  });
  it('allows one past the contiguous verified prefix', () => {
    expect(furthestReachable([0, 1])).toBe(2);
    expect(clampStep(4, [0, 1])).toBe(2);
    expect(clampStep(1, [0, 1])).toBe(1);
  });
  it('a gap blocks everything after it (deep links cannot skip a step)', () => {
    expect(furthestReachable([0, 2, 3])).toBe(1);
    expect(clampStep(3, [0, 2, 3])).toBe(1);
  });
  it('all verified reaches the last step, never beyond', () => {
    const all = [...Array(STEP_COUNT).keys()];
    expect(furthestReachable(all)).toBe(STEP_COUNT - 1);
    expect(clampStep(99, all)).toBe(STEP_COUNT - 1);
  });
  it('clamps negatives and non-integers to 0', () => {
    expect(clampStep(-3, [0, 1])).toBe(0);
    expect(clampStep(1.5, [0, 1])).toBe(0);
    expect(clampStep(NaN, [0, 1])).toBe(0);
  });
});

describe('progressPercent', () => {
  it('counts distinct in-range verified steps', () => {
    expect(progressPercent([])).toBe(0);
    expect(progressPercent([0, 1])).toBe(40);
    expect(progressPercent([0, 0, 1, 99, -1])).toBe(40);
    expect(progressPercent([0, 1, 2, 3, 4])).toBe(100);
  });
});
