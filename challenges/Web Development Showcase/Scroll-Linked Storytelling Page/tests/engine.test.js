import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import {
    ENGINES,
    STORAGE_KEY,
    cssTimelineSupported,
    otherEngine,
    readStoredEngine,
    resolveEngine,
    writeStoredEngine,
} from '../src/engine.js';

const memory = () => {
    const map = new Map();
    return {
        getItem: (k) => (map.has(k) ? map.get(k) : null),
        setItem: (k, v) => map.set(k, String(v)),
    };
};

const throwing = {
    getItem: () => {
        throw new DOMException('blocked', 'SecurityError');
    },
    setItem: () => {
        throw new DOMException('full', 'QuotaExceededError');
    },
};

describe('resolveEngine', () => {
    it('defaults to native CSS where supported and to GSAP where not', () => {
        expect(resolveEngine({ stored: null, cssSupported: true })).toBe('css');
        expect(resolveEngine({ stored: null, cssSupported: false })).toBe('gsap');
    });

    it('honours a stored choice, but never picks CSS when the browser cannot run it', () => {
        expect(resolveEngine({ stored: 'gsap', cssSupported: true })).toBe('gsap');
        expect(resolveEngine({ stored: 'css', cssSupported: true })).toBe('css');
        expect(resolveEngine({ stored: 'css', cssSupported: false })).toBe('gsap');
    });

    it('property: CSS is only ever chosen when supported, and the result is always an engine', () => {
        fc.assert(
            fc.property(fc.constantFrom(null, 'css', 'gsap', 'bogus'), fc.boolean(), (stored, cssSupported) => {
                const e = resolveEngine({ stored, cssSupported });
                expect(ENGINES).toContain(e);
                if (e === 'css') expect(cssSupported).toBe(true);
            }),
        );
    });
});

describe('storage', () => {
    it('round-trips a valid engine and ignores anything else', () => {
        const s = memory();
        expect(writeStoredEngine(s, 'gsap')).toBe(true);
        expect(readStoredEngine(s)).toBe('gsap');
        s.setItem(STORAGE_KEY, 'javascript:alert(1)');
        expect(readStoredEngine(s)).toBeNull();
    });

    it('treats blocked or full storage as "no choice" without throwing', () => {
        expect(readStoredEngine(throwing)).toBeNull();
        expect(readStoredEngine(null)).toBeNull();
        expect(writeStoredEngine(throwing, 'css')).toBe(false);
        expect(writeStoredEngine(null, 'css')).toBe(false);
    });
});

describe('cssTimelineSupported and otherEngine', () => {
    it('asks CSS.supports for animation-timeline, and reports false when that throws or is missing', () => {
        expect(cssTimelineSupported({ supports: (p, v) => p === 'animation-timeline' && v === 'view()' })).toBe(true);
        expect(cssTimelineSupported({ supports: () => false })).toBe(false);
        expect(
            cssTimelineSupported({
                supports: () => {
                    throw new Error('x');
                },
            }),
        ).toBe(false);
        expect(cssTimelineSupported(null)).toBe(false);
    });

    it('swaps the two engines', () => {
        expect(otherEngine('css')).toBe('gsap');
        expect(otherEngine('gsap')).toBe('css');
    });
});
