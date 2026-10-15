import { describe, expect, test } from 'vitest';
import { parseClientMsg } from '../server/schema.ts';

const parse = (value: unknown) => parseClientMsg(JSON.stringify(value));

const stroke = (over: object = {}) => ({
    t: 'stroke',
    id: 's1',
    color: '#aabbcc',
    size: 4,
    pts: [0, 0, 1, 1],
    ...over,
});
const add = (item: object) => ({ t: 'op', opId: 'o1', op: { k: 'add', item } });

describe('valid messages', () => {
    test('a stroke, a shape and a note parse', () => {
        expect(parse(add(stroke()))).not.toBeNull();
        expect(
            parse(add({ t: 'shape', k: 'arrow', id: 'a1', color: '#000000', size: 3, x1: 0, y1: 0, x2: 5, y2: 5 })),
        ).not.toBeNull();
        expect(
            parse(add({ t: 'text', id: 't1', color: '#fff176', size: 24, x: 1, y: 2, text: 'hi', sticky: true })),
        ).not.toBeNull();
    });

    test('a client-supplied author is dropped, not trusted', () => {
        const msg = parse(add(stroke({ author: 'mallory' })));
        expect(msg?.t === 'op' && msg.op.k === 'add' && 'author' in msg.op.item).toBe(false);
    });
});

describe('rejected messages', () => {
    test.each([
        ['not json', 'nope{'],
        ['unknown type', { t: 'rm -rf' }],
        ['a client trying to send res', { t: 'op', opId: 'o', op: { k: 'res', ids: ['a'] } }],
        ['bad colour', add(stroke({ color: 'red' }))],
        ['css injection in colour', add(stroke({ color: '#000000;background:url(x)' }))],
        ['odd number of coordinates', add(stroke({ pts: [0, 0, 1] }))],
        ['pressure count mismatch', add(stroke({ pr: [0.5] }))],
        ['pressure out of range', add(stroke({ pr: [2, 2] }))],
        ['NaN coordinate', '{"t":"cur","x":NaN,"y":1}'],
        ['coordinate out of range', { t: 'cur', x: 1e12, y: 0 }],
        ['too many points', add(stroke({ pts: new Array(10_002).fill(1) }))],
        ['huge stroke size', add(stroke({ size: 1e6 }))],
        ['zero size', add(stroke({ size: 0 }))],
        ['text too long', add({ t: 'text', id: 't', color: '#000000', size: 20, x: 0, y: 0, text: 'x'.repeat(2001), sticky: false })],
        ['id with a slash', { t: 'op', opId: '../x', op: { k: 'del', ids: ['a'] } }],
        ['empty id list', { t: 'op', opId: 'o', op: { k: 'del', ids: [] } }],
        ['too many ids', { t: 'op', opId: 'o', op: { k: 'del', ids: Array.from({ length: 2001 }, (_, i) => `i${i}`) } }],
        ['unknown shape kind', add({ t: 'shape', k: 'star', id: 'a', color: '#000000', size: 3, x1: 0, y1: 0, x2: 1, y2: 1 })],
        ['hello with a wrong version', { t: 'hello', v: 2, id: 'a', name: 'a', last: null }],
        ['hello with a negative seq', { t: 'hello', v: 1, id: 'a', name: 'a', last: -1 }],
    ])('%s', (_name, value) => {
        const raw = typeof value === 'string' ? value : JSON.stringify(value);
        expect(parseClientMsg(raw)).toBeNull();
    });

    test('a prototype-pollution payload is just another invalid message', () => {
        expect(parseClientMsg('{"t":"op","opId":"o","op":{"k":"del","ids":["a"]},"__proto__":{"x":1}}')).not.toBeNull();
        expect(({} as Record<string, unknown>).x).toBeUndefined();
    });
});
