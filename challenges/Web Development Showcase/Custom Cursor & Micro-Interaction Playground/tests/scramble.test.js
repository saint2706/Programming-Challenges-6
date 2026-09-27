import { describe, expect, it } from 'vitest';
import { scrambleStep } from '../scripts/scramble.js';

describe('scrambleStep', () => {
  const alwaysZero = () => 0;
  const alwaysNearOne = () => 0.999999;

  it('returns the full target once revealCount reaches the target length', () => {
    expect(scrambleStep('HELLO', 5, alwaysZero)).toBe('HELLO');
  });

  it('preserves the resolved prefix and only scrambles the remainder', () => {
    const result = scrambleStep('HELLO', 2, alwaysZero);
    expect(result.slice(0, 2)).toBe('HE');
    expect(result).toHaveLength(5);
  });

  it('keeps spaces intact instead of scrambling them', () => {
    const result = scrambleStep('AB CD', 0, alwaysZero);
    expect(result[2]).toBe(' ');
  });

  it('clamps an out-of-range revealCount instead of throwing', () => {
    expect(scrambleStep('HI', 99, alwaysZero)).toBe('HI');
    expect(scrambleStep('HI', -5, alwaysZero)).toHaveLength(2);
  });

  it('produces different scrambled characters for different random values', () => {
    const low = scrambleStep('XYZ', 0, alwaysZero);
    const high = scrambleStep('XYZ', 0, alwaysNearOne);
    expect(low).not.toBe(high);
  });
});
