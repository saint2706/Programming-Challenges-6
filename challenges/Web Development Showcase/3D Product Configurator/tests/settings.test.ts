import { describe, expect, test } from 'vitest';
import { LIMITS, defaults, guessDiagonal, parseSettings } from '../src/headcoupled/calibration.ts';

describe('guessing the size of the screen', () => {
    test('common laptop and desktop sizes come out right or one step off', () => {
        // [CSS width, CSS height, the real diagonal]: Windows laptops at 125-150% scaling, monitors at 100%.
        const cases: [number, number, number][] = [
            [1536, 864, 14],
            [1280, 720, 13.3],
            [1920, 1080, 21.5], // a 24 inch monitor: the nearest common size below it
            [2560, 1440, 27],
        ];
        for (const [w, h, real] of cases) {
            const guess = guessDiagonal(w, h);
            expect(Math.abs(guess - real) / real, `${w}x${h}`).toBeLessThan(0.16);
        }
    });

    test('a Mac laptop is known to run at 127 CSS pixels per inch, which makes its guess right', () => {
        expect(guessDiagonal(1470, 956, true)).toBe(14); // MacBook Air, 13.6 inch: the nearest common size
        expect(guessDiagonal(1512, 982, true)).toBe(14); // MacBook Pro 14 inch
        expect(guessDiagonal(1728, 1117, true)).toBe(16); // MacBook Pro 16 inch
        expect(guessDiagonal(1512, 982, false)).toBeGreaterThan(14); // without that knowledge it would say 16
    });

    test('the shape of the screen is read off it', () => {
        expect(defaults(1920, 1080).aspect).toBeCloseTo(16 / 9, 9);
        expect(defaults(1512, 982, true).aspect).toBeCloseTo(1512 / 982, 9);
    });
});

describe('settings saved in the browser (which anything could have written)', () => {
    const guess = defaults(1536, 864);

    test('good values replace the guesses, the shape of the screen is never taken from storage', () => {
        const s = parseSettings({ diagonal: 24, hfov: 78, depth: 1.5, aspect: 99 }, guess);
        expect(s).toMatchObject({ diagonal: 24, hfov: 78, depth: 1.5, aspect: guess.aspect });
    });

    test('values outside what makes sense are pulled to the nearest sensible one', () => {
        const s = parseSettings({ diagonal: 5000, hfov: -3, depth: 1e9 }, guess);
        expect(s.diagonal).toBe(LIMITS.diagonal[1]);
        expect(s.hfov).toBe(LIMITS.hfov[0]);
        expect(s.depth).toBe(LIMITS.depth[1]);
    });

    test.each([null, 5, 'x', [], { diagonal: 'big', hfov: NaN, depth: Infinity }, { diagonal: null }])('%j leaves the guesses alone', (raw) => {
        expect(parseSettings(raw, guess)).toEqual(guess);
    });
});
