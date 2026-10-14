import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { fuzzyMatch, highlightRuns, prepareText } from '../src/fuzzy.js';
import { allAlignments, scoreAlignment } from './oracle.js';

// A small alphabet that exercises every character class: lower, UPPER, digit, space, delimiter.
const ALPHABET = 'abAB1 -/_.';
const smallText = fc.string({ unit: fc.constantFrom(...ALPHABET), minLength: 1, maxLength: 9 });
const smallPattern = fc.string({ unit: fc.constantFrom(...'abAB1'), minLength: 1, maxLength: 3 });

describe('fuzzyMatch properties', () => {
    it('matches exactly when the pattern is a subsequence (brute-force oracle)', () => {
        fc.assert(
            fc.property(smallPattern, smallText, (pattern, text) => {
                expect(fuzzyMatch(pattern, text) !== null).toBe(allAlignments(pattern, text).length > 0);
            }),
            { numRuns: 1500 },
        );
    });

    it('returns an alignment whose positions spell the pattern and whose score is what the rules say', () => {
        fc.assert(
            fc.property(smallPattern, smallText, (pattern, text) => {
                const match = fuzzyMatch(pattern, text);
                fc.pre(match !== null);
                const chars = Array.from(text);
                expect(match.positions).toHaveLength(pattern.length);
                match.positions.forEach((at, i) => {
                    expect(chars[at].toLowerCase()).toBe(pattern[i].toLowerCase());
                    if (i > 0) expect(at).toBeGreaterThan(match.positions[i - 1]);
                });
                expect(match.score).toBe(scoreAlignment(match.positions, prepareText(text)));
            }),
            { numRuns: 1500 },
        );
    });

    it('never scores above the best alignment, and is optimal almost always', () => {
        let cases = 0;
        let optimal = 0;
        fc.assert(
            fc.property(smallPattern, smallText, (pattern, text) => {
                const match = fuzzyMatch(pattern, text);
                fc.pre(match !== null);
                const best = Math.max(...allAlignments(pattern, text).map((a) => a.score));
                expect(match.score).toBeLessThanOrEqual(best);
                cases += 1;
                if (match.score === best) optimal += 1;
            }),
            { numRuns: 3000 },
        );
        // The DP keeps one run-bonus per cell (like fzf), so a rare input can score below the true best.
        expect(optimal / cases).toBeGreaterThan(0.99);
    });

    it('ignores the case of the pattern', () => {
        fc.assert(
            fc.property(smallPattern, smallText, (pattern, text) => {
                expect(fuzzyMatch(pattern.toUpperCase(), text)).toEqual(fuzzyMatch(pattern.toLowerCase(), text));
            }),
            { numRuns: 500 },
        );
    });

    it('matches any text against itself, letter for letter', () => {
        fc.assert(
            fc.property(fc.string({ minLength: 1, maxLength: 30 }), (text) => {
                const match = fuzzyMatch(text, text);
                expect(match).not.toBeNull();
                expect(match.positions).toEqual(Array.from({ length: Array.from(text).length }, (_, i) => i));
            }),
            { numRuns: 500 },
        );
    });

    it('matches a subsequence picked out of any text', () => {
        fc.assert(
            fc.property(
                fc.string({ minLength: 1, maxLength: 60 }),
                fc.array(fc.nat(), { minLength: 1, maxLength: 8 }),
                (text, picks) => {
                    const chars = Array.from(text);
                    const indexes = [...new Set(picks.map((p) => p % chars.length))].sort((a, b) => a - b);
                    const pattern = indexes.map((i) => chars[i]).join('');
                    expect(fuzzyMatch(pattern, text)).not.toBeNull();
                },
            ),
            { numRuns: 500 },
        );
    });
});

describe('highlightRuns properties', () => {
    it('rebuilds the text, and marks exactly the given positions', () => {
        fc.assert(
            fc.property(
                fc.string({ maxLength: 40 }),
                fc.array(fc.nat(), { maxLength: 10 }),
                (text, picks) => {
                    const chars = Array.from(text);
                    const positions = chars.length ? [...new Set(picks.map((p) => p % chars.length))] : [];
                    const runs = highlightRuns(text, positions);
                    expect(runs.map((r) => r.text).join('')).toBe(text);
                    const marked = runs.filter((r) => r.match).flatMap((r) => Array.from(r.text)).length;
                    expect(marked).toBe(positions.length);
                    for (let i = 1; i < runs.length; i += 1) expect(runs[i].match).not.toBe(runs[i - 1].match);
                },
            ),
            { numRuns: 500 },
        );
    });
});
