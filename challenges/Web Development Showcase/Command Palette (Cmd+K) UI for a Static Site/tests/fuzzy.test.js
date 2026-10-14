import { describe, expect, it } from 'vitest';
import { highlightRuns, fuzzyMatch, MAX_PATTERN, MAX_TEXT } from '../src/fuzzy.js';

const pos = (pattern, text) => fuzzyMatch(pattern, text)?.positions;
const score = (pattern, text) => fuzzyMatch(pattern, text)?.score;

describe('matching', () => {
    it('finds a subsequence and reports where', () => {
        expect(pos('gs', 'Getting started')).toEqual([0, 8]);
        expect(pos('plt', 'Palette')).toEqual([0, 2, 4]);
    });

    it('is a match only when the letters are in order', () => {
        expect(fuzzyMatch('ptl', 'Palette')).toBeNull();
        expect(fuzzyMatch('xyz', 'Palette')).toBeNull();
        expect(fuzzyMatch('palettes', 'Palette')).toBeNull();
    });

    it('has no match for an empty pattern', () => {
        expect(fuzzyMatch('', 'anything')).toBeNull();
    });

    it('ignores case and diacritics', () => {
        expect(pos('CAFE', 'Café au lait')).toEqual([0, 1, 2, 3]);
        expect(pos('cafe', 'CAFÉ')).toEqual([0, 1, 2, 3]);
        expect(pos('uber', 'Über uns')).toEqual([0, 1, 2, 3]);
    });

    it('counts astral characters as one position', () => {
        expect(pos('b', '😀😀b')).toEqual([2]);
        expect(pos('😀b', 'a😀b')).toEqual([1, 2]);
    });

    it('matches symbols and spaces literally', () => {
        expect(pos('a/b', 'a/b/c')).toEqual([0, 1, 2]);
        expect(pos('a b', 'alpha beta')).toEqual([0, 5, 6]);
    });
});

describe('ranking', () => {
    it('prefers word starts to mid-word hits', () => {
        // "ar" in "bar/arch": the word-start "ar" beats the "a" in "bar" plus the "r"
        expect(pos('ar', 'bar/arch')).toEqual([4, 5]);
        expect(score('gs', 'Getting started')).toBeGreaterThan(score('gs', 'Debugging tests'));
    });

    it('prefers a camelCase hump to a mid-word letter', () => {
        expect(pos('usa', 'useAuthSession')).toEqual([0, 1, 3]);
        expect(score('as', 'AuthSession')).toBeGreaterThan(score('as', 'baseline'));
    });

    it('prefers a consecutive run to scattered letters', () => {
        expect(score('inst', 'Install')).toBeGreaterThan(score('inst', 'Is not stable'));
    });

    it('prefers a tighter match to a wider one', () => {
        expect(score('ab', 'xaby')).toBeGreaterThan(score('ab', 'xaqby'));
        expect(score('ab', 'xaqby')).toBeGreaterThan(score('ab', 'xaqqby'));
    });

    it('gives a prefix match more than the same letters later', () => {
        expect(score('key', 'Keyboard')).toBeGreaterThan(score('key', 'Hotkeys'));
    });

    it('counts a digit after letters as a hump', () => {
        expect(score('v2', 'api v2')).toBeGreaterThan(score('v2', 'abcv2'));
    });
});

describe('limits', () => {
    it('refuses patterns longer than the cap', () => {
        const text = 'a'.repeat(MAX_PATTERN + 5);
        expect(fuzzyMatch('a'.repeat(MAX_PATTERN + 1), text)).toBeNull();
        expect(fuzzyMatch('a'.repeat(MAX_PATTERN), text)).not.toBeNull();
    });

    it('still matches inside a text longer than the alignment window', () => {
        const text = `${'x'.repeat(MAX_TEXT + 10)} needle`;
        const match = fuzzyMatch('needle', text);
        expect(match.positions).toEqual(Array.from({ length: 6 }, (_, i) => MAX_TEXT + 11 + i));
        expect(fuzzyMatch('zzz', text)).toBeNull();
    });
});

describe('highlightRuns', () => {
    it('splits the text into matched and unmatched runs', () => {
        expect(highlightRuns('Getting started', [0, 8])).toEqual([
            { text: 'G', match: true },
            { text: 'etting ', match: false },
            { text: 's', match: true },
            { text: 'tarted', match: false },
        ]);
    });

    it('merges neighbours and keeps astral characters whole', () => {
        expect(highlightRuns('a😀bc', [1, 2])).toEqual([
            { text: 'a', match: false },
            { text: '😀b', match: true },
            { text: 'c', match: false },
        ]);
    });

    it('returns the text untouched with no positions', () => {
        expect(highlightRuns('abc', [])).toEqual([{ text: 'abc', match: false }]);
        expect(highlightRuns('', [])).toEqual([]);
    });
});
