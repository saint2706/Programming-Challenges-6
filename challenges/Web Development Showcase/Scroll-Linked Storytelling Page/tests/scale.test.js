import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { linePath, linearScale } from '../src/scale.js';

describe('linearScale', () => {
    it('maps the domain ends onto the range ends', () => {
        const s = linearScale([1880, 2025], [56, 776]);
        expect(s(1880)).toBeCloseTo(56);
        expect(s(2025)).toBeCloseTo(776);
    });

    it('property: preserves order (a larger input never maps to a smaller output)', () => {
        fc.assert(
            fc.property(
                fc.double({ min: -100, max: 100, noNaN: true }),
                fc.double({ min: -100, max: 100, noNaN: true }),
                (a, b) => {
                    const s = linearScale([-100, 100], [0, 800]);
                    if (a <= b) expect(s(a)).toBeLessThanOrEqual(s(b));
                },
            ),
        );
    });
});

describe('linePath', () => {
    it('writes one M then L per point', () => {
        expect(linePath([[1, 2], [3.25, 4]])).toBe('M1.0 2.0 L3.3 4.0');
    });

    it('property: one command per point, and the first is always M', () => {
        const point = fc.tuple(fc.double({ noNaN: true, min: 0, max: 800 }), fc.double({ noNaN: true, min: 0, max: 440 }));
        fc.assert(
            fc.property(fc.array(point, { minLength: 1, maxLength: 200 }), (pts) => {
                const commands = linePath(pts).match(/[ML]/g);
                expect(commands).toHaveLength(pts.length);
                expect(commands[0]).toBe('M');
                expect(commands.slice(1).every((c) => c === 'L')).toBe(true);
            }),
        );
    });
});
