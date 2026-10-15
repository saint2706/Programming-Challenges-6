import { describe, expect, test } from 'vitest';
import { Room, approxBytes, cleanName } from '../server/room.ts';
import { MAX_ITEMS_PER_ROOM, MAX_MEMBERS_PER_ROOM, MAX_OFFSET, MAX_ROOM_BYTES, MAX_STROKE_POINTS } from '../shared/limits.ts';
import type { Item } from '../shared/types.ts';
import { Recorder, join, stroke } from './helpers.ts';

/** Time moves 100 ms per look, so the per-connection rate limit never gets in the way of limits under test. */
function roomWithClock(): Room {
    let t = 0;
    return new Room('r', { now: () => (t += 100) });
}

const add = (room: Room, who: string, rec: Recorder, item: Item, opId = `op-${item.id}`) => room.receive(who, rec, { t: 'op', opId, op: { k: 'add', item } });
const bigStroke = (id: string, points: number): Item => ({ ...stroke(id), pts: Array.from({ length: points * 2 }, (_, i) => i % 1000) }) as Item;

describe('how much a room will hold', () => {
    test('the visible-item cap refuses an add, and erasing frees room again', () => {
        const room = roomWithClock();
        const rec = join(room, 'alice');
        for (let i = 0; i < MAX_ITEMS_PER_ROOM; i++) add(room, 'alice', rec, stroke(`s${i}`));
        expect(room.board.items.size).toBe(MAX_ITEMS_PER_ROOM);
        add(room, 'alice', rec, stroke('one-too-many'));
        expect(rec.ofType('nack').at(-1)).toMatchObject({ reason: 'full' });

        room.receive('alice', rec, { t: 'op', opId: 'erase', op: { k: 'del', ids: ['s0', 's1', 's2'] } });
        add(room, 'alice', rec, stroke('fits-now'));
        expect(room.board.items.has('fits-now')).toBe(true);
    });

    test('the byte budget refuses huge items even when the item count is small', () => {
        const room = roomWithClock();
        const rec = join(room, 'alice');
        const each = approxBytes(bigStroke('x', MAX_STROKE_POINTS));
        const fit = Math.floor(MAX_ROOM_BYTES / each);
        for (let i = 0; i < fit; i++) add(room, 'alice', rec, bigStroke(`b${i}`, MAX_STROKE_POINTS));
        expect(rec.ofType('nack')).toHaveLength(0);
        add(room, 'alice', rec, bigStroke('over', MAX_STROKE_POINTS));
        expect(rec.ofType('nack').at(-1)).toMatchObject({ reason: 'full' });
        expect(room.board.items.has('over')).toBe(false);
    });

    test('erased items are compacted away once no undo can bring them back, and not before', () => {
        const room = roomWithClock();
        const alice = join(room, 'alice');
        const bob = join(room, 'bob');
        add(room, 'alice', alice, stroke('mine'));
        room.receive('alice', alice, { t: 'op', opId: 'erase-mine', op: { k: 'del', ids: ['mine'] } });
        for (let i = 0; i < 1300; i++) add(room, 'bob', bob, stroke(`b${i}`));
        // Bob erases 1,200 strokes one at a time. His undo history keeps only the latest 200 steps,
        // so after 1,000 erasures the older ones can never be undone and are dropped.
        for (let i = 0; i < 1200; i++) room.receive('bob', bob, { t: 'op', opId: `e${i}`, op: { k: 'del', ids: [`b${i}`] } });

        expect(room.board.items.has('b0')).toBe(false);
        expect(room.board.items.has('b700')).toBe(false);
        expect(room.board.items.has('b900')).toBe(true); // still on Bob's undo stack
        expect(room.board.items.has('b1199')).toBe(true);
        expect(room.board.items.has('mine')).toBe(true); // Alice could still undo her erase
        for (let i = 1200; i < 1300; i++) expect(room.board.items.get(`b${i}`)?.deleted).toBe(false);
        expect(room.board.items.size).toBeLessThan(1 + 450 + 100);
        expect(room.board.items.size).toBeGreaterThan(1 + 400 + 100 - 5);

        // Undo still works for what was kept, and the saved form no longer carries what was dropped.
        room.receive('bob', bob, { t: 'undo', opId: 'u1' });
        expect(room.board.items.get('b1199')?.deleted).toBe(false);
        expect(room.toPersisted().items).toHaveLength(room.board.items.size);
    });

    test('the room accepts only so many people at once', () => {
        const room = new Room('r');
        for (let i = 0; i < MAX_MEMBERS_PER_ROOM; i++) join(room, `u${i}`);
        const late = join(room, 'one-more');
        expect(late.closed?.code).toBe(1013);
        expect(room.memberCount).toBe(MAX_MEMBERS_PER_ROOM);
        const again = join(room, 'u0'); // an existing person reconnecting still gets in
        expect(again.closed).toBeNull();
    });

    test('repeated moves cannot push an item beyond what a saved file can hold', () => {
        const room = roomWithClock();
        const rec = join(room, 'alice');
        add(room, 'alice', rec, stroke('s1'));
        const move = (i: number) => room.receive('alice', rec, { t: 'op', opId: `m${i}`, op: { k: 'mv', ids: ['s1'], dx: 1e6, dy: 0 } });
        for (let i = 0; i < 12; i++) move(i);
        const state = room.board.items.get('s1')!;
        expect(Math.abs(state.dx)).toBeLessThanOrEqual(MAX_OFFSET);
        expect(rec.ofType('nack').some((n) => n.reason === 'range')).toBe(true);
    });
});

describe('what a peer can make you hold or receive', () => {
    test('a live stroke stops being relayed after the longest stroke an honest client sends', () => {
        const room = roomWithClock();
        const alice = join(room, 'alice');
        const bob = join(room, 'bob');
        const chunk = { t: 'live', sid: 's', pts: Array.from({ length: 800 }, (_, i) => i), start: { color: '#000000', size: 4 } } as const;
        for (let i = 0; i < 30; i++) room.receive('alice', alice, i === 0 ? chunk : { t: 'live', sid: 's', pts: chunk.pts });
        const relayed = bob.ofType('live').reduce((n, m) => n + m.pts.length / 2, 0);
        expect(relayed).toBeLessThanOrEqual(MAX_STROKE_POINTS);
        expect(relayed).toBeGreaterThan(MAX_STROKE_POINTS - 400);
    });

    test('name changes, liveEnd and ping are rate limited, and an unchanged name is ignored', () => {
        const room = new Room('r', { now: () => 0 }); // frozen clock: the ephemeral bucket (burst 240) never refills
        const alice = join(room, 'alice');
        const bob = join(room, 'bob');
        for (let i = 0; i < 1000; i++) room.receive('alice', alice, { t: 'name', name: `Alice ${i}` });
        for (let i = 0; i < 1000; i++) room.receive('alice', alice, { t: 'liveEnd', sid: `s${i}` });
        for (let i = 0; i < 1000; i++) room.receive('alice', alice, { t: 'ping' });
        expect(bob.ofType('rename').length + bob.ofType('liveEnd').length + alice.ofType('pong').length).toBeLessThanOrEqual(240);

        const room2 = new Room('r');
        const a2 = join(room2, 'alice');
        const b2 = join(room2, 'bob');
        room2.receive('alice', a2, { t: 'name', name: 'Alice' });
        const before = b2.ofType('rename').length;
        room2.receive('alice', a2, { t: 'name', name: 'Alice' });
        expect(before).toBe(1);
        expect(b2.ofType('rename').length).toBe(before); // repeating the same name sends nothing
    });
});

describe('names', () => {
    test('zero-width, bidi-override and control characters are removed', () => {
        expect(cleanName('Al‮ice​⁦x⁩﻿')).toBe('Alicex');
        expect(cleanName('  \u0007Bob\u0000  ')).toBe('Bob');
        expect(cleanName('‮‮')).toBe('');
    });
});
