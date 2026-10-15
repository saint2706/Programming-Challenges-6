// @vitest-environment jsdom
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import fc from 'fast-check';
import { beforeAll, describe, expect, test } from 'vitest';

interface Format {
    format(value: unknown): string;
    formatArgs(args: unknown[]): string;
    limits: { MAX_STRING: number; MAX_TOTAL: number; MAX_ITEMS: number; MAX_DEPTH: number };
}

let f: Format;

beforeAll(() => {
    // The file the sandbox page loads, run as-is in this realm so `instanceof` agrees with the test's own values.
    new Function(readFileSync(join(process.cwd(), 'public', 'sandbox-format.js'), 'utf8'))();
    f = (globalThis as unknown as { __pgFormat: Format }).__pgFormat;
});

describe('values', () => {
    test.each([
        [undefined, 'undefined'],
        [null, 'null'],
        [true, 'true'],
        [42, '42'],
        [-0, '-0'],
        [NaN, 'NaN'],
        [Infinity, 'Infinity'],
        [10n, '10n'],
        [Symbol('s'), 'Symbol(s)'],
        ['plain', 'plain'],
        [new Date(0), '1970-01-01T00:00:00.000Z'],
        [new Date(NaN), 'Invalid Date'],
        [/a+/g, '/a+/g'],
        [[], '[]'],
        [{}, '{}'],
    ])('%s', (value, text) => {
        expect(f.format(value)).toBe(text);
    });

    test('nested strings are quoted, top-level ones are not', () => {
        expect(f.format(["it's", 'a\nb'])).toBe(`[ 'it\\'s', 'a\\nb' ]`);
    });

    test('functions and classes', () => {
        expect(f.format(function named() {})).toBe('[Function: named]');
        expect(f.format(() => {})).toMatch(/^\[Function/);
        expect(f.format(class Thing {})).toBe('[class Thing]');
    });

    test('objects: keys, nesting, class names, symbols, null prototype', () => {
        expect(f.format({ a: 1, 'b-c': 'x', nested: { d: [1, 2] } })).toBe(`{ a: 1, 'b-c': 'x', nested: { d: [ 1, 2 ] } }`);
        class Point { x = 1; y = 2; }
        expect(f.format(new Point())).toBe('Point { x: 1, y: 2 }');
        expect(f.format({ [Symbol('k')]: 1 })).toBe('{ [Symbol(k)]: 1 }');
        expect(f.format(Object.create(null))).toBe('[Object: null prototype] {}');
    });

    test('Map, Set and typed arrays', () => {
        expect(f.format(new Map([['a', 1]]))).toBe(`Map(1) { 'a' => 1 }`);
        expect(f.format(new Set([1, 2]))).toBe('Set(2) { 1, 2 }');
        expect(f.format(new Uint8Array([1, 2, 3]))).toBe('Uint8Array(3) [ 1, 2, 3 ]');
        expect(f.format(new ArrayBuffer(8))).toBe('ArrayBuffer { byteLength: 8 }');
        expect(f.format(new WeakMap())).toMatch(/WeakMap/);
        expect(f.format(Promise.resolve(1))).toMatch(/Promise/);
    });

    test('errors: the stack at the top, a short form inside', () => {
        const e = new TypeError('bad');
        expect(f.format(e)).toContain('TypeError: bad');
        expect(f.format({ e })).toBe('{ e: [TypeError: bad] }');
    });

    test('DOM nodes show as tags, not as their whole tree', () => {
        const el = document.createElement('div');
        el.id = 'x';
        el.className = 'a b';
        el.innerHTML = '<p>lots of children</p>';
        expect(f.format(el)).toBe('<div id="x" class="a b">');
        expect(f.format(document.createTextNode('t'))).toBe('[#text]');
    });
});

describe('it can never be tricked into doing too much or throwing', () => {
    test('circular references', () => {
        const a: Record<string, unknown> = { name: 'a' };
        a.self = a;
        const list: unknown[] = [1];
        list.push(list);
        expect(f.format(a)).toBe('{ name: \'a\', self: [Circular] }');
        expect(f.format(list)).toBe('[ 1, [Circular] ]');
    });

    test('depth, item and key limits', () => {
        const deep = { a: { b: { c: { d: { e: 1 } } } } };
        expect(f.format(deep)).toBe('{ a: { b: { c: [Object] } } }');
        expect(f.format(Array.from({ length: 500 }, (_, i) => i))).toContain('… 400 more items');
        const wide: Record<string, number> = {};
        for (let i = 0; i < 200; i++) wide[`k${i}`] = i;
        expect(f.format(wide)).toContain('… 150 more properties');
    });

    test('a huge string is cut and says how much was cut', () => {
        const text = f.format('x'.repeat(100_000));
        expect(text.length).toBeLessThan(f.limits.MAX_STRING + 100);
        expect(text).toMatch(/more characters/);
    });

    test('getters are shown, never run (a getter could loop forever or throw)', () => {
        let ran = false;
        const o = {
            get trap() {
                ran = true;
                throw new Error('should not run');
            },
            set only(_: number) {},
            plain: 1,
        };
        expect(f.format(o)).toBe('{ trap: [Getter], only: [Setter], plain: 1 }');
        expect(ran).toBe(false);
    });

    test('a revoked proxy, a throwing toString and a hostile prototype are survivable', () => {
        const revocable = Proxy.revocable({}, {});
        revocable.revoke();
        expect(f.format(revocable.proxy)).toBe('[Uninspectable]');
        const hostile = new Proxy({}, { ownKeys() { throw new Error('no'); }, getPrototypeOf() { throw new Error('no'); } });
        expect(f.format(hostile)).toBe('[Uninspectable]');
        expect(f.format({ toString() { throw new Error('no'); } })).toContain('toString');
    });

    test('total output is capped however it is built', () => {
        const wideAndDeep = Array.from({ length: 100 }, () => Array.from({ length: 100 }, () => 'x'.repeat(150)));
        expect(f.format(wideAndDeep).length).toBeLessThan(f.limits.MAX_TOTAL * 3);
        expect(f.formatArgs([wideAndDeep, wideAndDeep, wideAndDeep]).length).toBeLessThanOrEqual(f.limits.MAX_TOTAL + 100);
    });

    test('never throws on arbitrary data', () => {
        fc.assert(fc.property(fc.anything({ withMap: true, withSet: true, withBigInt: true, withTypedArray: true, withNullPrototype: true, withObjectString: true }), (v) => {
            expect(typeof f.format(v)).toBe('string');
        }));
    });
});

describe('console.log-style arguments', () => {
    test.each([
        [['a', 1, { b: 2 }], `a 1 { b: 2 }`],
        [['%s is %d', 'x', 5], 'x is 5'],
        [['%i and %f', '7.9', '1.5'], '7 and 1.5'],
        [['%j', { a: 1 }], '{"a":1}'],
        [['%o', [1]], '[ 1 ]'],
        [['100%', 'done'], '100% done'],
        [['%c styled', 'color: red'], ' styled'],
        [['%d%%', 50], '50%'],
        [['%s'], '%s'],
        [['%s %s', 'only one'], 'only one %s'],
        [[], ''],
    ])('%j', (args, text) => {
        expect(f.formatArgs(args as unknown[])).toBe(text);
    });

    test('%j on a circular value does not throw', () => {
        const a: Record<string, unknown> = {};
        a.a = a;
        expect(f.formatArgs(['%j', a])).toBe('[Circular]');
    });
});
