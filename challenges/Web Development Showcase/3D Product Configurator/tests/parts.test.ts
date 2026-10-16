import { describe, expect, test } from 'vitest';
import { FINISH_IDS, FINISHES, PALETTE, PART_IDS, PART_LABELS, PRESETS, defaultDesign, normalizeHex, parseDesign, presetOf } from '../src/parts.ts';

describe('colours', () => {
    test.each([
        ['#ABCDEF', '#abcdef'],
        ['abcdef', '#abcdef'],
        ['#abc', '#aabbcc'],
        ['  #0F0 ', '#00ff00'],
    ])('%s is accepted as %s', (input, out) => expect(normalizeHex(input)).toBe(out));

    test.each(['', '#12', '#12345', '#1234567', 'red', 'rgb(0,0,0)', '#ggg', '#abcdef; background: url(x)', 5, null, undefined, {}])('%j is not a colour', (input) => expect(normalizeHex(input)).toBeNull());
});

describe('the parts and designs', () => {
    test('every part has a name, and every preset and the palette are valid', () => {
        for (const id of PART_IDS) expect(PART_LABELS[id].length).toBeGreaterThan(0);
        for (const p of PRESETS) for (const id of PART_IDS) expect(normalizeHex(p.colors[id]), `${p.id} ${id}`).toBe(p.colors[id]);
        for (const s of PALETTE) expect(normalizeHex(s.hex)).toBe(s.hex);
        expect(new Set(PALETTE.map((s) => s.name)).size).toBe(PALETTE.length);
        expect(new Set(PRESETS.map((p) => p.id)).size).toBe(PRESETS.length);
    });

    test('the finishes keep roughness between 0 and 1 and the clearcoat too, and matte is the model as the artist made it', () => {
        for (const id of FINISH_IDS) {
            const f = FINISHES[id];
            expect(f.roughness).toBeGreaterThan(0);
            expect(f.roughness).toBeLessThanOrEqual(1);
            expect(f.clearcoat).toBeGreaterThanOrEqual(0);
            expect(f.clearcoat).toBeLessThanOrEqual(1);
        }
        expect(FINISHES.matte.roughness).toBe(1);
    });

    test('the default design is the first preset, and is recognised as it', () => {
        expect(presetOf(defaultDesign().colors)).toBe(PRESETS[0]!.id);
        const d = defaultDesign();
        d.colors.laces = '#123456';
        expect(presetOf(d.colors)).toBeNull();
    });

    test('a default design is a fresh object every time, so editing one never changes the next', () => {
        const a = defaultDesign();
        a.colors.upper = '#000000';
        expect(defaultDesign().colors.upper).not.toBe('#000000');
        expect(PRESETS[0]!.colors.upper).not.toBe('#000000');
    });
});

describe('a saved design from storage (which anything could have written)', () => {
    test('a good one comes back as it was', () => {
        const d = defaultDesign();
        d.colors.upper = '#aa00ff';
        d.finishes.upper = 'patent';
        expect(parseDesign(JSON.parse(JSON.stringify(d)))).toEqual(d);
    });

    test('a bad part is replaced by its default and the good ones are kept', () => {
        const d = parseDesign({ colors: { upper: 'url(javascript:alert(1))', laces: '#123456', sole: '#ffffff' }, finishes: { upper: 'chrome', laces: 'gloss' } })!;
        expect(d.colors.upper).toBe(defaultDesign().colors.upper);
        expect(d.colors.laces).toBe('#123456');
        expect(d.finishes.upper).toBe(defaultDesign().finishes.upper);
        expect(d.finishes.laces).toBe('gloss');
        expect(Object.keys(d.colors).sort()).toEqual([...PART_IDS].sort());
    });

    test.each([null, 5, 'x', [], { colors: 5 }, { colors: null }, { finishes: {} }])('%j is no design', (raw) => {
        expect(parseDesign(raw)).toBeNull();
    });

    test('prototype keys are not finishes', () => {
        const d = parseDesign({ colors: {}, finishes: { upper: 'constructor', laces: '__proto__', accents: 'toString' } })!;
        expect(d.finishes.upper).toBe(defaultDesign().finishes.upper);
        expect(d.finishes.laces).toBe(defaultDesign().finishes.laces);
        expect(d.finishes.accents).toBe(defaultDesign().finishes.accents);
    });
});
