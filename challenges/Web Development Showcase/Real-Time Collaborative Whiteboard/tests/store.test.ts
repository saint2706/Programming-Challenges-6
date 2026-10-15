import { mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import { Bucket } from '../server/bucket.ts';
import { Room, type Persisted } from '../server/room.ts';
import { Rooms } from '../server/rooms.ts';
import { RoomStore } from '../server/store.ts';
import { join as joinRoom, stroke } from './helpers.ts';

let dir: string;
beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'wb-'));
});
afterEach(() => rmSync(dir, { recursive: true, force: true }));

function saved(id: string): Persisted {
    const room = new Room(id);
    const rec = joinRoom(room, 'alice');
    room.receive('alice', rec, { t: 'op', opId: 'o1', op: { k: 'add', item: stroke('s1') } });
    return room.toPersisted();
}

describe('RoomStore', () => {
    test('round-trips a room', async () => {
        const store = new RoomStore(dir);
        await store.save('demo', saved('demo'));
        const loaded = store.load('demo');
        expect(loaded?.seq).toBe(1);
        expect(loaded?.items[0]?.item).toMatchObject({ id: 's1', author: 'alice' });
    });

    test('hostile room ids never become paths', async () => {
        const store = new RoomStore(dir);
        for (const id of ['../../evil', 'a/b', 'CON', 'NUL', '..', 'x'.repeat(300)]) {
            await store.save(id, { ...saved('t'), id });
        }
        const files = readdirSync(dir);
        expect(files).toHaveLength(6);
        expect(files.every((f) => /^[0-9a-f]{32}\.json$/.test(f))).toBe(true);
    });

    test('a corrupt file is moved aside and the room starts empty', async () => {
        const store = new RoomStore(dir);
        await store.save('demo', saved('demo'));
        const [file] = readdirSync(dir);
        writeFileSync(join(dir, file!), '{"v":1,"id":"demo","seq":3,"items":[{"item":{"t":"stroke"},"deleted":false,"dx":0,"dy":0}]}');
        expect(store.load('demo')).toBeNull();
        expect(readdirSync(dir).some((f) => f.includes('.corrupt-'))).toBe(true);
    });

    test('a file saved under another id is not loaded as this room', async () => {
        const store = new RoomStore(dir);
        await store.save('a', saved('a'));
        const [file] = readdirSync(dir);
        // Same bytes, but the id inside disagrees with the name the hash was made from.
        writeFileSync(join(dir, file!), readFileSync(join(dir, file!), 'utf8').replace('"id":"a"', '"id":"b"'));
        expect(store.load('a')).toBeNull();
    });

    test('concurrent saves land in order, so the newest wins', async () => {
        const store = new RoomStore(dir);
        const first = { ...saved('demo'), seq: 1 };
        const second = { ...saved('demo'), seq: 2 };
        await Promise.all([store.save('demo', first), store.save('demo', second)]);
        expect(store.load('demo')?.seq).toBe(2);
        expect(readdirSync(dir).filter((f) => f.endsWith('.tmp'))).toEqual([]);
    });
});

describe('Rooms', () => {
    // Only timers are faked: the file writes are real and finish on their own.
    beforeEach(() => vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] }));
    afterEach(() => vi.useRealTimers());

    test('saves shortly after a change, once per burst', async () => {
        const store = new RoomStore(dir);
        const save = vi.spyOn(store, 'save').mockResolvedValue();
        const rooms = new Rooms(store, { saveDelayMs: 100 });
        const opened = rooms.open('demo');
        if ('error' in opened) throw new Error(opened.error);
        const rec = joinRoom(opened.room, 'alice');
        for (let i = 0; i < 5; i++) {
            opened.room.receive('alice', rec, { t: 'op', opId: `o${i}`, op: { k: 'add', item: stroke(`s${i}`) } });
        }
        expect(save).not.toHaveBeenCalled();
        await vi.advanceTimersByTimeAsync(150);
        expect(save).toHaveBeenCalledTimes(1);
        rooms.dispose();
    });

    test('an empty room is flushed and unloaded after the idle time', async () => {
        const store = new RoomStore(dir);
        const rooms = new Rooms(store, { idleMs: 1000, saveDelayMs: 10_000 });
        const opened = rooms.open('demo');
        if ('error' in opened) throw new Error(opened.error);
        const rec = joinRoom(opened.room, 'alice');
        opened.room.receive('alice', rec, { t: 'op', opId: 'o1', op: { k: 'add', item: stroke('s1') } });
        opened.room.leave('alice', rec);
        rooms.release('demo');
        await vi.advanceTimersByTimeAsync(1500);
        while (rooms.loaded > 0) await new Promise((r) => setImmediate(r));
        expect(rooms.loaded).toBe(0);
        expect(store.load('demo')?.seq).toBe(1); // flushed on the way out, not lost
        rooms.dispose();
    });

    test('a room someone rejoined is not unloaded', async () => {
        const rooms = new Rooms(new RoomStore(dir), { idleMs: 1000 });
        const opened = rooms.open('demo');
        if ('error' in opened) throw new Error(opened.error);
        const rec = joinRoom(opened.room, 'alice');
        opened.room.leave('alice', rec);
        rooms.release('demo');
        joinRoom(opened.room, 'bob');
        await vi.advanceTimersByTimeAsync(2000);
        expect(rooms.loaded).toBe(1);
        rooms.dispose();
    });

    test('caps on rooms in memory and rooms on disk', async () => {
        const store = new RoomStore(dir);
        const rooms = new Rooms(store, { maxLoaded: 2, maxSaved: 1 });
        expect('room' in rooms.open('a')).toBe(true);
        expect('room' in rooms.open('b')).toBe(true);
        expect(rooms.open('c')).toEqual({ error: 'busy' });
        await store.save('x', saved('x'));
        const second = new Rooms(store, { maxSaved: 1 });
        expect(second.open('brand-new')).toEqual({ error: 'full' });
        expect('room' in second.open('x')).toBe(true); // an existing room still opens
        rooms.dispose();
        second.dispose();
    });
});

describe('Bucket', () => {
    test('allows a burst, then refills over time', () => {
        let t = 0;
        const bucket = new Bucket(10, 3, () => t);
        expect([bucket.take(), bucket.take(), bucket.take(), bucket.take()]).toEqual([true, true, true, false]);
        t = 100; // one token back
        expect([bucket.take(), bucket.take()]).toEqual([true, false]);
        t = 10_000; // never above the burst
        expect([bucket.take(), bucket.take(), bucket.take(), bucket.take()]).toEqual([true, true, true, false]);
    });
});
