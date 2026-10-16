import fc from 'fast-check';
import { describe, expect, test } from 'vitest';
import { HEAD_FILTER, OneEuroFilter, OneEuroVector3 } from '../src/headcoupled/filter.ts';

/** A deterministic stand-in for tracking noise. */
function noise(seed: number) {
    let s = seed;
    return () => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 2 ** 32 - 0.5;
    };
}

const FPS = 30;

describe('the 1-euro filter', () => {
    test('the first sample passes through, and a constant stays constant', () => {
        const f = new OneEuroFilter();
        expect(f.filter(3.5, 0)).toBe(3.5);
        for (let i = 1; i < 100; i++) expect(f.filter(3.5, i / FPS)).toBeCloseTo(3.5, 12);
    });

    test('when the signal is still it removes most of the noise', () => {
        const f = new OneEuroFilter();
        const n = noise(1);
        let rawVar = 0;
        let outVar = 0;
        const sigma = 0.004; // 4 mm of tracking jitter
        for (let i = 0; i < 600; i++) {
            const raw = sigma * (n() * 2);
            const out = f.filter(raw, i / FPS);
            if (i > 60) {
                rawVar += raw * raw;
                outVar += out * out;
            }
        }
        expect(Math.sqrt(outVar / rawVar)).toBeLessThan(0.45); // less than half the jitter remains
    });

    test('when the signal moves it follows with little lag, far less than a heavy plain average', () => {
        const filter = new OneEuroFilter();
        let ema = 0;
        const smoothing = 0.15; // an average heavy enough to calm the same jitter
        let lagFilter = 0;
        let lagEma = 0;
        for (let i = 0; i < 90; i++) {
            const t = i / FPS;
            const truth = i < 30 ? 0 : (i - 30) * 0.01; // stands still, then moves at 30 cm/s
            const out = filter.filter(truth, t);
            ema += smoothing * (truth - ema);
            if (i > 60) {
                lagFilter = Math.max(lagFilter, truth - out);
                lagEma = Math.max(lagEma, truth - ema);
            }
        }
        expect(lagFilter).toBeLessThan(lagEma / 3);
        expect(lagFilter).toBeLessThan(0.02); // under 2 cm behind at 30 cm/s
    });

    test('a step settles to the new value, and faster the bigger the jump', () => {
        const settle = (size: number) => {
            const f = new OneEuroFilter();
            f.filter(0, 0);
            for (let i = 1; i < 300; i++) {
                if (Math.abs(f.filter(size, i / FPS) - size) < size * 0.02) return i;
            }
            return 300;
        };
        expect(settle(0.3)).toBeLessThan(300);
        expect(settle(0.3)).toBeLessThan(settle(0.02));
    });

    test('odd timestamps are handled: repeats, going backwards, and a long gap', () => {
        const f = new OneEuroFilter();
        f.filter(1, 1);
        expect(f.filter(5, 1)).toBe(1); // same instant: no change
        expect(f.filter(5, 0.5)).toBe(1); // earlier: ignored
        const after = f.filter(5, 10); // a long pause
        expect(after).toBeGreaterThan(1);
        expect(Number.isFinite(after)).toBe(true);
    });

    test('reset forgets the past; a non-finite sample does not poison the state', () => {
        const f = new OneEuroFilter();
        f.filter(1, 0);
        f.filter(2, 0.1);
        f.reset();
        expect(f.filter(9, 5)).toBe(9);
        const g = new OneEuroFilter();
        g.filter(1, 0);
        for (let i = 1; i < 20; i++) expect(Number.isFinite(g.filter(1 + i * 0.01, i / FPS))).toBe(true);
    });

    test('never overshoots the range of what it has been given', () => {
        fc.assert(
            fc.property(fc.array(fc.double({ min: -1, max: 1, noNaN: true }), { minLength: 2, maxLength: 80 }), (values) => {
                const f = new OneEuroFilter();
                const lo = Math.min(...values);
                const hi = Math.max(...values);
                values.forEach((v, i) => {
                    const out = f.filter(v, i / FPS);
                    expect(out).toBeGreaterThanOrEqual(lo - 1e-9);
                    expect(out).toBeLessThanOrEqual(hi + 1e-9);
                });
            }),
        );
    });

    test('the default settings are the ones that were tuned', () => {
        expect(HEAD_FILTER).toEqual({ minCutoff: 1.2, beta: 8, derivativeCutoff: 1 });
    });
});

test('the vector filter smooths each axis on its own', () => {
    const f = new OneEuroVector3();
    f.filter({ x: 0, y: 0, z: 0.6 }, 0);
    const out = f.filter({ x: 0.1, y: 0, z: 0.6 }, 1 / FPS);
    expect(out.x).toBeGreaterThan(0);
    expect(out.x).toBeLessThan(0.1);
    expect(out.y).toBe(0);
    expect(out.z).toBeCloseTo(0.6, 12);
    f.reset();
    expect(f.filter({ x: 5, y: 5, z: 5 }, 9)).toEqual({ x: 5, y: 5, z: 5 });
});
