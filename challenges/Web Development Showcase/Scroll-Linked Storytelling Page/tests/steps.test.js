import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { pickActiveStep, stepForProgress } from '../src/steps.js';

const entries = (flags) => flags.map((isIntersecting, i) => ({ id: `s${i}`, isIntersecting }));

describe('pickActiveStep', () => {
    it('keeps the current step while it is still crossing the middle line', () => {
        expect(pickActiveStep(entries([true, true, false]), 's1')).toBe('s1');
    });

    it('moves to the first crossing step when the current one has left', () => {
        expect(pickActiveStep(entries([false, true, false]), 's0')).toBe('s1');
    });

    it('keeps the previous value when nothing crosses (for example while scrolling past the story)', () => {
        expect(pickActiveStep(entries([false, false]), 's1')).toBe('s1');
        expect(pickActiveStep(entries([false, false]), null)).toBeNull();
    });

    it('property: a crossing step wins whenever the previous one is not crossing', () => {
        fc.assert(
            fc.property(
                fc.array(fc.boolean(), { minLength: 1, maxLength: 8 }),
                fc.option(fc.integer({ min: 0, max: 7 }), { nil: null }),
                (flags, prev) => {
                    const es = entries(flags);
                    const current = prev === null ? null : `s${prev}`;
                    const next = pickActiveStep(es, current);
                    const crossing = es.filter((e) => e.isIntersecting).map((e) => e.id);
                    if (crossing.length === 0) expect(next).toBe(current);
                    else if (current !== null && crossing.includes(current)) expect(next).toBe(current);
                    else expect(next).toBe(crossing[0]);
                },
            ),
        );
    });
});

describe('stepForProgress', () => {
    it('maps 0 to the first step and 1 to the last', () => {
        expect(stepForProgress(0, 5)).toBe(0);
        expect(stepForProgress(1, 5)).toBe(4);
    });

    it('property: monotone in progress and always a valid index; bad input is clamped', () => {
        fc.assert(
            fc.property(
                fc.double({ min: 0, max: 1, noNaN: true }),
                fc.double({ min: 0, max: 1, noNaN: true }),
                fc.integer({ min: 1, max: 12 }),
                (a, b, count) => {
                    const lo = Math.min(a, b);
                    const hi = Math.max(a, b);
                    const i = stepForProgress(lo, count);
                    const j = stepForProgress(hi, count);
                    expect(i).toBeLessThanOrEqual(j);
                    expect(j).toBeGreaterThanOrEqual(0);
                    expect(j).toBeLessThan(count);
                },
            ),
        );
        expect(stepForProgress(Number.NaN, 5)).toBe(0);
        expect(stepForProgress(7, 5)).toBe(4);
        expect(() => stepForProgress(0, 0)).toThrow(RangeError);
    });
});
