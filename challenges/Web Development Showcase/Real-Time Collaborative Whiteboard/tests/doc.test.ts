import { describe, expect, test, vi } from 'vitest';
import type { ClientMsg, ServerMsg, SnapshotItem } from '../shared/types.ts';
import { Doc } from '../src/doc.ts';
import { stroke, textItem } from './helpers.ts';

function setup() {
    const sent: ClientMsg[] = [];
    let n = 0;
    let clock = 0;
    const doc = new Doc((m) => sent.push(m), () => `id${n++}`, 'me', () => clock);
    doc.handle({ t: 'welcome', you: { id: 'me', name: 'Me', color: '#e6194b' }, seq: 0, mode: 'snapshot', items: [], users: [{ id: 'me', name: 'Me', color: '#e6194b' }], canUndo: false, canRedo: false });
    return { doc, sent, tick: (ms: number) => (clock += ms) };
}

const snap = (...items: SnapshotItem[]) => items;
const entry = (item: ReturnType<typeof stroke>): SnapshotItem => ({ item, deleted: false, dx: 0, dy: 0 });
const op = (seq: number, by: string, opId: string, o: Extract<ServerMsg, { t: 'op' }>['op']): ServerMsg => ({ t: 'op', seq, by, opId, op: o });

describe('your own changes show at once, before the server confirms', () => {
    test('an added item is visible and waiting, then confirmed in place', () => {
        const { doc, sent } = setup();
        const id = doc.submit({ k: 'add', item: stroke('s1') });
        expect(doc.visible().map((v) => v.item.id)).toEqual(['s1']);
        expect(doc.pendingCount).toBe(1);
        expect(sent.at(-1)).toMatchObject({ t: 'op', opId: id });

        doc.handle(op(1, 'me', id, { k: 'add', item: { ...stroke('s1'), author: 'me' } }));
        expect(doc.visible()).toHaveLength(1); // not shown twice
        expect(doc.pendingCount).toBe(0);
        expect(doc.seq).toBe(1);
    });

    test('an erase hides the item immediately and a nack brings it back', () => {
        const { doc } = setup();
        doc.handle({ t: 'welcome', you: { id: 'me', name: 'Me', color: '#000000' }, seq: 1, mode: 'snapshot', items: snap(entry(stroke('s1'))), users: [], canUndo: false, canRedo: false });
        const nacks: string[] = [];
        doc.onNack = (r) => nacks.push(r);
        const id = doc.submit({ k: 'del', ids: ['s1'] });
        expect(doc.visible()).toHaveLength(0);
        doc.handle({ t: 'nack', opId: id, reason: 'rate' });
        expect(doc.visible()).toHaveLength(1);
        expect(nacks).toEqual(['rate']);
        expect(doc.pendingCount).toBe(0);
    });

    test('a pending move and a pending edit are layered over the confirmed item', () => {
        const { doc } = setup();
        doc.handle({ t: 'welcome', you: { id: 'me', name: 'Me', color: '#000000' }, seq: 2, mode: 'snapshot', items: snap({ item: textItem('t1', 'one'), deleted: false, dx: 5, dy: 0 }), users: [], canUndo: false, canRedo: false });
        doc.submit({ k: 'mv', ids: ['t1'], dx: 10, dy: 3 });
        doc.submit({ k: 'edit', id: 't1', text: 'two' });
        const v = doc.find('t1')!;
        expect([v.dx, v.dy]).toEqual([15, 3]);
        expect(v.item.t === 'text' && v.item.text).toBe('two');
    });

    test("someone else's op on the same item does not undo your pending edit", () => {
        const { doc } = setup();
        doc.handle({ t: 'welcome', you: { id: 'me', name: 'Me', color: '#000000' }, seq: 1, mode: 'snapshot', items: snap({ item: textItem('t1', 'one'), deleted: false, dx: 0, dy: 0 }), users: [], canUndo: false, canRedo: false });
        const mine = doc.submit({ k: 'edit', id: 't1', text: 'mine' });
        doc.handle(op(2, 'bob', 'b1', { k: 'edit', id: 't1', text: 'bobs' }));
        expect((doc.find('t1')!.item as { text: string }).text).toBe('mine'); // yours is still in flight, and will land last
        doc.handle(op(3, 'me', mine, { k: 'edit', id: 't1', text: 'mine' }));
        expect((doc.find('t1')!.item as { text: string }).text).toBe('mine');
        expect(doc.pendingCount).toBe(0);
    });
});

describe('the server stream', () => {
    test('an op that repeats a seq already applied is ignored', () => {
        const { doc } = setup();
        doc.handle(op(1, 'bob', 'b1', { k: 'add', item: stroke('s1') }));
        doc.handle(op(1, 'bob', 'b1', { k: 'mv', ids: ['s1'], dx: 99, dy: 99 })); // would corrupt if applied
        expect(doc.find('s1')).toMatchObject({ dx: 0, dy: 0 });
    });

    test('undo and redo are requests: nothing changes until the server answers', () => {
        const { doc, sent } = setup();
        doc.handle(op(1, 'me', 'x', { k: 'add', item: stroke('s1') }));
        doc.undo();
        expect(doc.visible()).toHaveLength(1);
        expect(sent.at(-1)).toMatchObject({ t: 'undo' });
        expect(doc.pendingCount).toBe(1);
        doc.handle({ t: 'undoState', canUndo: false, canRedo: true });
        expect([doc.canUndo, doc.canRedo]).toEqual([false, true]);
    });

    test('a welcome replaces the board and re-sends what was still pending', () => {
        const { doc, sent } = setup();
        doc.submit({ k: 'add', item: stroke('mine') });
        sent.length = 0;
        doc.disconnected();
        expect(doc.synced).toBe(false);
        doc.handle({ t: 'welcome', you: { id: 'me', name: 'Me', color: '#000000' }, seq: 9, mode: 'snapshot', items: snap(entry(stroke('theirs'))), users: [], canUndo: false, canRedo: false });
        expect(doc.visible().map((v) => v.item.id)).toEqual(['theirs', 'mine']); // theirs, then your unsent stroke on top
        expect(sent).toHaveLength(1);
        expect(sent[0]).toMatchObject({ t: 'op' });
    });

    test('hello asks for a catch-up only when there is something to catch up from', () => {
        const { doc } = setup();
        expect(doc.hello('Me')).toMatchObject({ last: 0 });
        doc.handle(op(4, 'bob', 'b', { k: 'add', item: stroke('s') }));
        expect(doc.hello('Me')).toMatchObject({ last: 4 });
        const fresh = new Doc(() => {}, () => 'x', 'me');
        expect(fresh.hello('Me')).toMatchObject({ last: null });
    });
});

describe('live strokes, cursors and people', () => {
    test('a live stroke accumulates chunks, and the committed stroke replaces it', () => {
        const { doc } = setup();
        doc.handle({ t: 'live', by: 'bob', sid: 's9', pts: [0, 0, 1, 1], start: { color: '#000000', size: 4 } });
        doc.handle({ t: 'live', by: 'bob', sid: 's9', pts: [2, 2] });
        expect([...doc.live.values()][0]!.pts).toEqual([0, 0, 1, 1, 2, 2]);
        doc.handle(op(1, 'bob', 'b', { k: 'add', item: stroke('s9') }));
        expect(doc.live.size).toBe(0);
    });

    test('a peer cannot grow a live stroke past the longest an honest client sends', () => {
        const { doc } = setup();
        const chunk = Array.from({ length: 800 }, (_, i) => i);
        doc.handle({ t: 'live', by: 'bob', sid: 's', pts: chunk, start: { color: '#000000', size: 4 } });
        for (let i = 0; i < 30; i++) doc.handle({ t: 'live', by: 'bob', sid: 's', pts: chunk });
        expect([...doc.live.values()][0]!.pts.length / 2).toBeLessThanOrEqual(4000);
    });

    test('a chunk whose start was lost is dropped rather than drawn from nowhere', () => {
        const { doc } = setup();
        doc.handle({ t: 'live', by: 'bob', sid: 's9', pts: [2, 2] });
        expect(doc.live.size).toBe(0);
    });

    test('liveEnd, leave and disconnect clear what belonged to them', () => {
        const { doc } = setup();
        doc.handle({ t: 'join', user: { id: 'bob', name: 'Bob', color: '#00ff00' } });
        doc.handle({ t: 'cur', by: 'bob', x: 1, y: 2 });
        doc.handle({ t: 'live', by: 'bob', sid: 's', pts: [0, 0], start: { color: '#000000', size: 4 } });
        doc.handle({ t: 'liveEnd', by: 'bob', sid: 's' });
        expect(doc.live.size).toBe(0);
        doc.handle({ t: 'leave', id: 'bob' });
        expect(doc.users.has('bob')).toBe(false);
        expect(doc.cursors.has('bob')).toBe(false);
    });

    test('a rename updates the user, and your own name', () => {
        const { doc } = setup();
        doc.handle({ t: 'rename', id: 'me', name: 'Ada' });
        expect(doc.me?.name).toBe('Ada');
        expect(doc.users.get('me')?.name).toBe('Ada');
    });

    test('onChange fires, and boardVersion moves only for changes to the drawing', () => {
        const { doc } = setup();
        const changed = vi.fn();
        doc.onChange = changed;
        const board = doc.boardVersion;
        doc.handle({ t: 'cur', by: 'bob', x: 1, y: 1 });
        expect(changed).toHaveBeenCalledTimes(1);
        expect(doc.boardVersion).toBe(board);
        doc.submit({ k: 'add', item: stroke('s') });
        expect(doc.boardVersion).toBeGreaterThan(board);
    });
});
