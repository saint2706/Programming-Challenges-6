import fc from 'fast-check';
import { expect, test } from 'vitest';
import { describe as describeBoard, Net, stroke, textItem } from './helpers.ts';

/**
 * The property the whole design rests on: whatever order messages are delivered in, however often
 * tabs drop and reconnect, and even if the server restarts, once the network is quiet every tab
 * shows exactly what the server has.
 */
type Cmd =
    | { c: 'add'; tab: number; text: boolean }
    | { c: 'del'; tab: number; pick: number; count: number }
    | { c: 'mv'; tab: number; pick: number; dx: number; dy: number }
    | { c: 'edit'; tab: number; pick: number; text: string }
    | { c: 'undo' | 'redo'; tab: number }
    | { c: 'up' | 'down' | 'drop' | 'connect'; tab: number }
    | { c: 'flush' }
    | { c: 'restart' };

const TABS = 3;
const tab = fc.integer({ min: 0, max: TABS - 1 });
const pick = fc.nat(50);

const cmd: fc.Arbitrary<Cmd> = fc.oneof(
    { weight: 24, arbitrary: fc.record({ c: fc.constant('add' as const), tab, text: fc.boolean() }) },
    {
        weight: 10,
        arbitrary: fc.record({ c: fc.constant('del' as const), tab, pick, count: fc.integer({ min: 1, max: 3 }) }),
    },
    {
        weight: 14,
        arbitrary: fc.record({
            c: fc.constant('mv' as const),
            tab,
            pick,
            dx: fc.integer({ min: -9, max: 9 }),
            dy: fc.integer({ min: -9, max: 9 }),
        }),
    },
    {
        weight: 6,
        arbitrary: fc.record({ c: fc.constant('edit' as const), tab, pick, text: fc.constantFrom('a', 'b', 'c', '') }),
    },
    { weight: 12, arbitrary: fc.record({ c: fc.constantFrom('undo' as const, 'redo' as const), tab }) },
    { weight: 40, arbitrary: fc.record({ c: fc.constantFrom('up' as const, 'down' as const), tab }) },
    { weight: 6, arbitrary: fc.record({ c: fc.constant('drop' as const), tab }) },
    { weight: 8, arbitrary: fc.record({ c: fc.constant('connect' as const), tab }) },
    { weight: 1, arbitrary: fc.record({ c: fc.constant('restart' as const) }) },
    { weight: 14, arbitrary: fc.record({ c: fc.constant('flush' as const) }) },
);

function run(cmds: Cmd[]): Net {
    const net = new Net(TABS);
    net.settle(); // everyone starts connected and in sync, so reconnects below are real catch-ups
    let n = 0;
    for (const x of cmds) {
        if (x.c === 'restart') {
            net.restart();
            continue;
        }
        if (x.c === 'flush') {
            net.flush();
            continue;
        }
        const doc = net.docs[x.tab]!;
        switch (x.c) {
            case 'add':
                doc.submit({
                    k: 'add',
                    item: x.text ? textItem(`i${n++}`, 'start') : stroke(`i${n++}`, n, n),
                });
                break;
            case 'del':
            case 'mv': {
                const seen = doc.visible();
                if (seen.length === 0) break;
                const ids = Array.from(
                    { length: x.c === 'del' ? x.count : 1 },
                    (_, k) => seen[(x.pick + k) % seen.length]!.item.id,
                );
                doc.submit(x.c === 'del' ? { k: 'del', ids } : { k: 'mv', ids, dx: x.dx, dy: x.dy });
                break;
            }
            case 'edit': {
                const texts = doc.visible().filter((v) => v.item.t === 'text');
                const target = texts[x.pick % texts.length];
                if (target) doc.submit({ k: 'edit', id: target.item.id, text: x.text });
                break;
            }
            case 'undo':
                doc.undo();
                break;
            case 'redo':
                doc.redo();
                break;
            case 'up':
                net.deliverUp(x.tab);
                break;
            case 'down':
                net.deliverDown(x.tab);
                break;
            case 'drop':
                net.disconnect(x.tab);
                break;
            case 'connect':
                net.connect(x.tab);
                break;
        }
    }
    net.settle();
    return net;
}

function assertConverged(net: Net): void {
    const server = describeBoard(net.room.board);
    for (const [i, doc] of net.docs.entries()) {
        expect(doc.pendingCount, `tab ${i} still has pending ops`).toBe(0);
        expect(doc.seq, `tab ${i} seq`).toBe(net.room.seq);
        expect(describeBoard(doc.confirmed), `tab ${i} board`).toEqual(server);
        // What is drawn is the confirmed state, nothing extra from the optimistic layer.
        const shown = doc.visible().map((v) => [v.item.id, v.dx, v.dy]);
        const truth = net.room.board
            .visible()
            .map((s) => [s.item.id, s.dx, s.dy]);
        expect(shown, `tab ${i} visible`).toEqual(truth);
    }
}

test('every tab ends with the server state, under reordering, drops, reconnects and restarts', () => {
    fc.assert(
        fc.property(fc.array(cmd, { minLength: 60, maxLength: 220, size: 'max' }), (cmds) => {
            assertConverged(run(cmds));
        }),
        { numRuns: 400 },
    );
});

test('undo is per user: undoing never removes someone else\'s stroke', () => {
    fc.assert(
        fc.property(fc.array(cmd, { minLength: 60, maxLength: 160, size: 'max' }), (cmds) => {
            const net = run(cmds);
            // Any stroke a tab drew and still can see was not removed by another tab's undo, so asking
            // tab 1 to undo everything it can leaves every stroke that tab 0 drew (and kept) alone.
            const keptByZero = new Set(
                net.room.board
                    .visible()
                    .filter((s) => s.item.author === 'c0')
                    .map((s) => s.item.id),
            );
            for (let k = 0; k < 400; k++) net.docs[1]!.undo();
            net.settle();
            for (const id of keptByZero) {
                expect(net.room.board.items.get(id)?.deleted, `tab 0's ${id} survived tab 1's undos`).toBe(false);
            }
            assertConverged(net);
        }),
        { numRuns: 120 },
    );
});
