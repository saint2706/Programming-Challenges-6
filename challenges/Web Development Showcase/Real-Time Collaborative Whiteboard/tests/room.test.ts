import { describe, expect, test } from 'vitest';
import { Room } from '../server/room.ts';
import type { ClientMsg } from '../shared/types.ts';
import { join, Recorder, stroke, textItem } from './helpers.ts';

function setup() {
    const room = new Room('r', { now: () => 0 });
    const a = join(room, 'alice');
    const b = join(room, 'bob');
    return { room, a, b };
}

const send = (room: Room, who: string, rec: Recorder, msg: Parameters<Room['receive']>[2]) =>
    room.receive(who, rec, msg);

describe('joining', () => {
    test('a new tab gets a snapshot and everyone else hears about it', () => {
        const room = new Room('r');
        const a = join(room, 'alice');
        const b = join(room, 'bob');
        const welcome = b.ofType('welcome')[0]!;
        expect(welcome.mode).toBe('snapshot');
        expect(welcome.users.map((u) => u.id).sort()).toEqual(['alice', 'bob']);
        expect(a.ofType('join')[0]!.user.id).toBe('bob');
    });

    test('collaborators get distinct colours while the palette lasts', () => {
        const room = new Room('r');
        const colors = ['a', 'b', 'c', 'd'].map((id) => join(room, id).ofType('welcome')[0]!.you.color);
        expect(new Set(colors).size).toBe(4);
    });

    test('a returning tab with a recent seq gets only the ops it missed', () => {
        const { room, a } = setup();
        send(room, 'alice', a, { t: 'op', opId: 'o1', op: { k: 'add', item: stroke('s1') } });
        send(room, 'alice', a, { t: 'op', opId: 'o2', op: { k: 'add', item: stroke('s2') } });
        const back = join(room, 'carol', 1);
        const welcome = back.ofType('welcome')[0]!;
        expect(welcome.mode).toBe('catchup');
        expect(welcome.ops!.map((e) => e.seq)).toEqual([2]);
    });

    test('a tab that is ahead of the server (it lost data) gets a snapshot', () => {
        const room = new Room('r');
        const rec = join(room, 'alice', 99);
        expect(rec.ofType('welcome')[0]!.mode).toBe('snapshot');
    });

    test('a second connection with the same id replaces the first', () => {
        const room = new Room('r');
        const first = join(room, 'alice');
        const second = join(room, 'alice');
        expect(first.closed?.code).toBe(4001);
        expect(second.closed).toBeNull();
        // The stale socket cannot act any more.
        room.receive('alice', first, { t: 'op', opId: 'o', op: { k: 'add', item: stroke('s') } });
        expect(room.seq).toBe(0);
    });

    test('names lose control characters and are capped', () => {
        const room = new Room('r');
        const rec = new Recorder();
        room.join(rec, { t: 'hello', v: 1, id: 'x', name: `  Ann\u0007\u0000${'z'.repeat(60)}`, last: null });
        const name = rec.ofType('welcome')[0]!.you.name;
        expect(name.startsWith('Ann')).toBe(true);
        expect(name.length).toBeLessThanOrEqual(24);
        expect(name).not.toMatch(/[\u0000-\u001f]/);
    });
});

describe('ops', () => {
    test('an add is sequenced, stamped with its author and broadcast to everyone', () => {
        const { room, a, b } = setup();
        send(room, 'alice', a, {
            t: 'op',
            opId: 'o1',
            op: { k: 'add', item: { ...stroke('s1'), author: 'mallory' } },
        });
        for (const rec of [a, b]) {
            const op = rec.ofType('op')[0]!;
            expect(op).toMatchObject({ seq: 1, by: 'alice', opId: 'o1' });
            expect(op.op.k === 'add' && op.op.item.author).toBe('alice'); // the client's claim is ignored
        }
    });

    test('resending an op does not apply it twice', () => {
        const { room, a } = setup();
        const msg: ClientMsg = { t: 'op', opId: 'o1', op: { k: 'mv', ids: ['s1'], dx: 5, dy: 5 } };
        send(room, 'alice', a, { t: 'op', opId: 'o0', op: { k: 'add', item: stroke('s1') } });
        send(room, 'alice', a, msg);
        send(room, 'alice', a, msg);
        expect(room.board.items.get('s1')).toMatchObject({ dx: 5, dy: 5 });
        expect(a.ofType('ack')).toEqual([{ t: 'ack', opId: 'o1', seq: 2 }]);
        expect(room.seq).toBe(2);
    });

    test('an add with a taken id, and ops on unknown items, are refused', () => {
        const { room, a } = setup();
        send(room, 'alice', a, { t: 'op', opId: 'o1', op: { k: 'add', item: stroke('s1') } });
        send(room, 'alice', a, { t: 'op', opId: 'o2', op: { k: 'add', item: stroke('s1') } });
        send(room, 'alice', a, { t: 'op', opId: 'o3', op: { k: 'mv', ids: ['nope'], dx: 1, dy: 1 } });
        send(room, 'alice', a, { t: 'op', opId: 'o4', op: { k: 'edit', id: 's1', text: 'x' } }); // not a text item
        expect(a.ofType('nack').map((n) => [n.opId, n.reason])).toEqual([
            ['o2', 'dup'],
            ['o3', 'unknown'],
            ['o4', 'unknown'],
        ]);
        expect(room.seq).toBe(1);
    });

    test('erasing something already erased is acknowledged but not sequenced again', () => {
        const { room, a, b } = setup();
        send(room, 'alice', a, { t: 'op', opId: 'o1', op: { k: 'add', item: stroke('s1') } });
        send(room, 'alice', a, { t: 'op', opId: 'o2', op: { k: 'del', ids: ['s1'] } });
        send(room, 'bob', b, { t: 'op', opId: 'o3', op: { k: 'del', ids: ['s1'] } });
        expect(room.seq).toBe(2);
        expect(b.ofType('ack')).toEqual([{ t: 'ack', opId: 'o3', seq: 2 }]);
    });

    test('moves add up instead of overwriting each other', () => {
        const { room, a, b } = setup();
        send(room, 'alice', a, { t: 'op', opId: 'o1', op: { k: 'add', item: stroke('s1') } });
        send(room, 'alice', a, { t: 'op', opId: 'o2', op: { k: 'mv', ids: ['s1'], dx: 10, dy: 0 } });
        send(room, 'bob', b, { t: 'op', opId: 'o3', op: { k: 'mv', ids: ['s1'], dx: 0, dy: 7 } });
        expect(room.board.items.get('s1')).toMatchObject({ dx: 10, dy: 7 });
    });

    test('the item cap and the per-connection rate limit answer with a nack', () => {
        const room = new Room('r', { now: () => 0 }); // a frozen clock: the bucket never refills
        const rec = join(room, 'alice');
        for (let i = 0; i < 70; i++) {
            room.receive('alice', rec, { t: 'op', opId: `o${i}`, op: { k: 'add', item: stroke(`s${i}`) } });
        }
        expect(room.seq).toBe(60); // the burst
        expect(rec.ofType('nack').every((n) => n.reason === 'rate')).toBe(true);
        expect(rec.ofType('nack')).toHaveLength(10);
    });

    test('edit changes text and an unchanged edit is a no-op', () => {
        const { room, a } = setup();
        send(room, 'alice', a, { t: 'op', opId: 'o1', op: { k: 'add', item: textItem('t1', 'one') } });
        send(room, 'alice', a, { t: 'op', opId: 'o2', op: { k: 'edit', id: 't1', text: 'two' } });
        send(room, 'alice', a, { t: 'op', opId: 'o3', op: { k: 'edit', id: 't1', text: 'two' } });
        const state = room.board.items.get('t1')!;
        expect(state.item.t === 'text' && state.item.text).toBe('two');
        expect(room.seq).toBe(2);
    });
});

describe('undo and redo', () => {
    test('undo removes your own last stroke and redo brings it back', () => {
        const { room, a, b } = setup();
        send(room, 'alice', a, { t: 'op', opId: 'o1', op: { k: 'add', item: stroke('s1') } });
        send(room, 'alice', a, { t: 'undo', opId: 'u1' });
        expect(room.board.items.get('s1')!.deleted).toBe(true);
        expect(b.ofType('op').at(-1)!.op).toEqual({ k: 'del', ids: ['s1'] });
        send(room, 'alice', a, { t: 'redo', opId: 'r1' });
        expect(room.board.items.get('s1')!.deleted).toBe(false);
    });

    test('undo only touches your own history', () => {
        const { room, a, b } = setup();
        send(room, 'alice', a, { t: 'op', opId: 'o1', op: { k: 'add', item: stroke('mine') } });
        send(room, 'bob', b, { t: 'op', opId: 'o2', op: { k: 'add', item: stroke('theirs') } });
        send(room, 'alice', a, { t: 'undo', opId: 'u1' });
        expect(room.board.items.get('mine')!.deleted).toBe(true);
        expect(room.board.items.get('theirs')!.deleted).toBe(false);
        send(room, 'alice', a, { t: 'undo', opId: 'u2' }); // nothing left of alice's
        expect(a.ofType('nack').at(-1)).toMatchObject({ opId: 'u2', reason: 'empty' });
        expect(room.board.items.get('theirs')!.deleted).toBe(false);
    });

    test('undoing an erase restores the strokes, undoing a move moves them back', () => {
        const { room, a } = setup();
        send(room, 'alice', a, { t: 'op', opId: 'o1', op: { k: 'add', item: stroke('s1') } });
        send(room, 'alice', a, { t: 'op', opId: 'o2', op: { k: 'mv', ids: ['s1'], dx: 30, dy: 0 } });
        send(room, 'alice', a, { t: 'op', opId: 'o3', op: { k: 'del', ids: ['s1'] } });
        send(room, 'alice', a, { t: 'undo', opId: 'u1' });
        expect(room.board.items.get('s1')).toMatchObject({ deleted: false, dx: 30 });
        send(room, 'alice', a, { t: 'undo', opId: 'u2' });
        expect(room.board.items.get('s1')).toMatchObject({ dx: 0 });
    });

    test('a new action ends the redo branch', () => {
        const { room, a } = setup();
        send(room, 'alice', a, { t: 'op', opId: 'o1', op: { k: 'add', item: stroke('s1') } });
        send(room, 'alice', a, { t: 'undo', opId: 'u1' });
        send(room, 'alice', a, { t: 'op', opId: 'o2', op: { k: 'add', item: stroke('s2') } });
        send(room, 'alice', a, { t: 'redo', opId: 'r1' });
        expect(a.ofType('nack').at(-1)).toMatchObject({ opId: 'r1', reason: 'empty' });
        expect(a.ofType('undoState').at(-1)).toEqual({ t: 'undoState', canUndo: true, canRedo: false });
    });

    test('undo skips entries that someone else already made moot', () => {
        const { room, a, b } = setup();
        send(room, 'alice', a, { t: 'op', opId: 'o1', op: { k: 'add', item: stroke('s1') } });
        send(room, 'alice', a, { t: 'op', opId: 'o2', op: { k: 'add', item: stroke('s2') } });
        send(room, 'bob', b, { t: 'op', opId: 'o3', op: { k: 'del', ids: ['s2'] } }); // bob erased s2
        send(room, 'alice', a, { t: 'undo', opId: 'u1' });
        // Undoing "add s2" would change nothing, so the undo reaches "add s1" instead.
        expect(room.board.items.get('s1')!.deleted).toBe(true);
    });

    test('a resent undo is applied once', () => {
        const { room, a } = setup();
        send(room, 'alice', a, { t: 'op', opId: 'o1', op: { k: 'add', item: stroke('s1') } });
        send(room, 'alice', a, { t: 'op', opId: 'o2', op: { k: 'add', item: stroke('s2') } });
        send(room, 'alice', a, { t: 'undo', opId: 'u1' });
        send(room, 'alice', a, { t: 'undo', opId: 'u1' });
        expect(room.board.items.get('s1')!.deleted).toBe(false);
        expect(room.board.items.get('s2')!.deleted).toBe(true);
    });

    test('an edit can be undone to the previous text', () => {
        const { room, a } = setup();
        send(room, 'alice', a, { t: 'op', opId: 'o1', op: { k: 'add', item: textItem('t1', 'one') } });
        send(room, 'alice', a, { t: 'op', opId: 'o2', op: { k: 'edit', id: 't1', text: 'two' } });
        send(room, 'alice', a, { t: 'undo', opId: 'u1' });
        const item = room.board.items.get('t1')!.item;
        expect(item.t === 'text' && item.text).toBe('one');
    });
});

describe('ephemeral messages', () => {
    test('live strokes and cursors reach the others but not the sender, and are never sequenced', () => {
        const { room, a, b } = setup();
        send(room, 'alice', a, {
            t: 'live',
            sid: 's1',
            pts: [1, 2],
            start: { color: '#000000', size: 4 },
        });
        send(room, 'alice', a, { t: 'cur', x: 5, y: 6 });
        expect(b.ofType('live')).toHaveLength(1);
        expect(b.ofType('cur')[0]).toMatchObject({ by: 'alice', x: 5, y: 6 });
        expect(a.ofType('live')).toHaveLength(0);
        expect(room.seq).toBe(0);
    });

    test('a collaborator who leaves mid-stroke has the stroke cleared for everyone', () => {
        const { room, a, b } = setup();
        send(room, 'alice', a, { t: 'live', sid: 's1', pts: [1, 2], start: { color: '#000000', size: 4 } });
        room.leave('alice', a);
        expect(b.ofType('liveEnd')).toEqual([{ t: 'liveEnd', by: 'alice', sid: 's1' }]);
        expect(b.ofType('leave')[0]).toEqual({ t: 'leave', id: 'alice' });
    });

    test('a rename reaches everyone', () => {
        const { room, a, b } = setup();
        send(room, 'alice', a, { t: 'name', name: 'Alice B.' });
        expect(b.ofType('rename')[0]).toEqual({ t: 'rename', id: 'alice', name: 'Alice B.' });
    });
});

test('a room restored from its saved form keeps items, tombstones and offsets', () => {
    const { room, a } = setup();
    send(room, 'alice', a, { t: 'op', opId: 'o1', op: { k: 'add', item: stroke('s1') } });
    send(room, 'alice', a, { t: 'op', opId: 'o2', op: { k: 'mv', ids: ['s1'], dx: 3, dy: 4 } });
    send(room, 'alice', a, { t: 'op', opId: 'o3', op: { k: 'add', item: stroke('s2') } });
    send(room, 'alice', a, { t: 'op', opId: 'o4', op: { k: 'del', ids: ['s2'] } });
    const restored = Room.restore(JSON.parse(JSON.stringify(room.toPersisted())));
    expect(restored.seq).toBe(4);
    expect(restored.board.items.get('s1')).toMatchObject({ dx: 3, dy: 4, deleted: false });
    expect(restored.board.items.get('s2')!.deleted).toBe(true);
});
