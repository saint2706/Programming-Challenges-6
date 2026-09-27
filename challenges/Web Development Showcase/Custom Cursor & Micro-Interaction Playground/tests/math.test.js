import { describe, expect, it } from 'vitest';
import { clamp, lerp, magneticOffset, tiltAngles } from '../scripts/math.js';

describe('clamp', () => {
  it('passes values already inside the range through unchanged', () => {
    expect(clamp(5, 0, 10)).toBe(5);
  });

  it('clamps below the minimum', () => {
    expect(clamp(-3, 0, 10)).toBe(0);
  });

  it('clamps above the maximum', () => {
    expect(clamp(42, 0, 10)).toBe(10);
  });

  it('swaps a reversed min/max instead of misbehaving', () => {
    expect(clamp(5, 10, 0)).toBe(5);
    expect(clamp(-1, 10, 0)).toBe(0);
  });
});

describe('lerp', () => {
  it('returns the start value at t=0', () => {
    expect(lerp(10, 20, 0)).toBe(10);
  });

  it('returns the end value at t=1', () => {
    expect(lerp(10, 20, 1)).toBe(20);
  });

  it('returns the midpoint at t=0.5', () => {
    expect(lerp(10, 20, 0.5)).toBe(15);
  });

  it('extrapolates for t outside 0..1', () => {
    expect(lerp(0, 10, 2)).toBe(20);
  });
});

describe('magneticOffset', () => {
  const center = { x: 100, y: 100 };

  it('returns zero offset when the cursor is outside the radius', () => {
    const cursor = { x: 300, y: 300 };
    expect(magneticOffset(cursor, center, 50, 0.5)).toEqual({ x: 0, y: 0 });
  });

  it('returns zero offset exactly at the radius boundary edge case of radius<=0', () => {
    const cursor = { x: 100, y: 100 };
    expect(magneticOffset(cursor, center, 0, 0.5)).toEqual({ x: 0, y: 0 });
  });

  it('scales the raw offset by strength when inside the radius', () => {
    const cursor = { x: 110, y: 100 };
    expect(magneticOffset(cursor, center, 50, 0.5)).toEqual({ x: 5, y: 0 });
  });

  it('handles diagonal offsets inside the radius', () => {
    const cursor = { x: 130, y: 140 };
    const result = magneticOffset(cursor, center, 100, 1);
    expect(result.x).toBeCloseTo(30);
    expect(result.y).toBeCloseTo(40);
  });
});

describe('tiltAngles', () => {
  const rect = { left: 0, top: 0, width: 200, height: 100 };

  it('returns zero rotation when the pointer is at the exact center', () => {
    expect(tiltAngles({ x: 100, y: 50 }, rect, 15)).toEqual({ rotateX: 0, rotateY: 0 });
  });

  it('tilts toward positive rotateY at the right edge', () => {
    const { rotateY } = tiltAngles({ x: 200, y: 50 }, rect, 15);
    expect(rotateY).toBeCloseTo(15);
  });

  it('tilts toward negative rotateY at the left edge', () => {
    const { rotateY } = tiltAngles({ x: 0, y: 50 }, rect, 15);
    expect(rotateY).toBeCloseTo(-15);
  });

  it('tilts toward positive rotateX at the top edge', () => {
    const { rotateX } = tiltAngles({ x: 100, y: 0 }, rect, 15);
    expect(rotateX).toBeCloseTo(15);
  });

  it('tilts toward negative rotateX at the bottom edge', () => {
    const { rotateX } = tiltAngles({ x: 100, y: 100 }, rect, 15);
    expect(rotateX).toBeCloseTo(-15);
  });

  it('clamps pointer positions outside the rect instead of over-rotating', () => {
    const result = tiltAngles({ x: -500, y: -500 }, rect, 15);
    expect(result.rotateX).toBeCloseTo(15);
    expect(result.rotateY).toBeCloseTo(-15);
  });

  it('returns zero rotation for a degenerate zero-size rect', () => {
    expect(tiltAngles({ x: 10, y: 10 }, { left: 0, top: 0, width: 0, height: 0 }, 15)).toEqual({
      rotateX: 0,
      rotateY: 0,
    });
  });
});
