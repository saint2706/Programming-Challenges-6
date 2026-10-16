import { beforeEach, describe, expect, test, vi } from 'vitest';
import { defaultDesign } from '../src/parts.ts';
import { Studio } from '../src/studio.svelte.ts';

const KEY = 'configurator:design:v1';

function storage(initial: Record<string, string> = {}) {
    const data = new Map(Object.entries(initial));
    const api = { getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => void data.set(k, v), removeItem: (k: string) => void data.delete(k), data };
    vi.stubGlobal('localStorage', api);
    return api;
}

beforeEach(() => {
    vi.unstubAllGlobals();
});

describe('the design the person is making', () => {
    test('starts from the default, and remembers every change', () => {
        const store = storage();
        const studio = new Studio();
        expect(studio.design).toEqual(defaultDesign());
        studio.setColor('laces', '#FF0000');
        studio.setFinish('upper', 'gloss');
        expect(JSON.parse(store.data.get(KEY)!)).toMatchObject({ colors: { laces: '#ff0000' }, finishes: { upper: 'gloss' } });
        expect(new Studio().design.colors.laces).toBe('#ff0000');
    });

    test('a colour that is not one is ignored, not stored', () => {
        const store = storage();
        const studio = new Studio();
        studio.setColor('upper', 'red; x');
        expect(studio.design.colors.upper).toBe(defaultDesign().colors.upper);
        expect(store.data.has(KEY)).toBe(false);
    });

    test('a preset replaces the colours but keeps the finishes', () => {
        storage();
        const studio = new Studio();
        studio.setFinish('laces', 'patent');
        studio.applyPreset('volt');
        expect(studio.design.colors.upper).toBe('#1f1f23');
        expect(studio.design.finishes.laces).toBe('patent');
        studio.applyPreset('no-such-preset');
        expect(studio.design.colors.upper).toBe('#1f1f23');
    });

    test('reset goes back to the default and is remembered', () => {
        const store = storage();
        const studio = new Studio();
        studio.applyPreset('ember');
        studio.setFinish('upper', 'patent');
        studio.reset();
        expect(studio.design).toEqual(defaultDesign());
        expect(JSON.parse(store.data.get(KEY)!)).toEqual(defaultDesign());
    });

    test.each(['{not json', 'null', '5', '[]', '{"colors":5}'])('corrupt saved data %s is ignored', (raw) => {
        storage({ [KEY]: raw });
        expect(new Studio().design).toEqual(defaultDesign());
    });

    test('storage that throws (blocked, or full) never stops the page', () => {
        vi.stubGlobal('localStorage', {
            getItem: () => {
                throw new DOMException('blocked', 'SecurityError');
            },
            setItem: () => {
                throw new DOMException('full', 'QuotaExceededError');
            },
        });
        const studio = new Studio();
        expect(() => studio.setColor('upper', '#123456')).not.toThrow();
        expect(studio.design.colors.upper).toBe('#123456'); // it still shows
    });
});
