import fc from 'fast-check';
import { describe, expect, test } from 'vitest';
import { parseSandboxEvent } from '../src/messages.ts';

describe('what the preview may say', () => {
    test('known messages are normalized', () => {
        expect(parseSandboxEvent({ t: 'ready' })).toEqual({ t: 'ready' });
        expect(parseSandboxEvent({ t: 'log', run: 3, level: 'warn', group: 2, text: 'hi', stack: 's' })).toEqual({ t: 'log', run: 3, level: 'warn', group: 2, text: 'hi', stack: 's' });
        expect(parseSandboxEvent({ t: 'scroll', run: 1, x: 10.7, y: 20 })).toEqual({ t: 'scroll', run: 1, x: 10, y: 20 });
        expect(parseSandboxEvent({ t: 'csp', run: 1, directive: 'script-src', blocked: 'inline', sample: 'x' })).toMatchObject({ t: 'csp', directive: 'script-src' });
    });

    test.each([null, undefined, 5, 'ready', [], { t: 'nope' }, { t: 'log', level: 'fatal' }, { t: 'log', level: 5 }, { t: 7 }, {}])('%j is ignored', (value) => {
        expect(parseSandboxEvent(value)).toBeNull();
    });

    test('sizes and numbers are clamped, so a hostile page cannot flood or confuse the playground', () => {
        const log = parseSandboxEvent({ t: 'log', run: 1, level: 'log', group: 9999, text: 'x'.repeat(1e6), stack: 'y'.repeat(1e6) });
        expect(log).toMatchObject({ group: 10 });
        expect(log && log.t === 'log' && log.text.length).toBe(20_000);
        expect(log && log.t === 'log' && log.stack.length).toBe(8000);
        const err = parseSandboxEvent({ t: 'error', run: -5, line: Infinity, column: NaN, message: 'm'.repeat(1e5), name: '' });
        expect(err).toMatchObject({ run: 0, line: 0, column: 0, name: 'Error' });
        expect(err && err.t === 'error' && err.message.length).toBe(2000);
        expect(parseSandboxEvent({ t: 'scroll', run: 1, x: -50, y: 1e300 })).toMatchObject({ x: 0, y: 10_000_000 });
    });

    test('non-string fields become empty text rather than objects that could be rendered', () => {
        const log = parseSandboxEvent({ t: 'log', run: 1, level: 'log', group: 0, text: { toString: () => '<img src=x onerror=alert(1)>' }, stack: ['x'] });
        expect(log).toMatchObject({ text: '', stack: '' });
    });

    test('never throws, and only ever returns a known type', () => {
        fc.assert(
            fc.property(fc.anything(), (value) => {
                const parsed = parseSandboxEvent(value);
                if (parsed) expect(['ready', 'clear', 'log', 'error', 'csp', 'scroll']).toContain(parsed.t);
            }),
        );
    });
});
