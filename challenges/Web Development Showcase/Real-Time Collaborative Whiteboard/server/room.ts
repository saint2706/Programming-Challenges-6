import { Board } from '../shared/board.ts';
import {
    MAX_ITEMS_PER_ROOM,
    MAX_MEMBERS_PER_ROOM,
    MAX_NAME,
    MAX_OFFSET,
    MAX_ROOM_BYTES,
    MAX_STROKE_POINTS,
} from '../shared/limits.ts';
import type {
    ClientMsg,
    ClientOp,
    Item,
    LogEntry,
    Op,
    ServerMsg,
    SnapshotItem,
    User,
} from '../shared/types.ts';
import { Bucket } from './bucket.ts';

/** One live connection as the room sees it. The WebSocket glue and the tests both implement this. */
export interface Conn {
    /** `json` is `msg` already serialized, shared by every recipient of a broadcast. */
    send(msg: ServerMsg, json?: string): void;
    close(code: number, reason: string): void;
}

export interface Persisted {
    v: 1;
    id: string;
    seq: number;
    items: SnapshotItem[];
}

interface UndoEntry {
    undo: Op;
    redo: Op;
}

interface Member {
    conn: Conn;
    user: User;
    ops: Bucket;
    ephemeral: Bucket;
    /** Live strokes in progress -> points relayed so far (capped, so a peer cannot grow your buffer forever). */
    liveSids: Map<string, number>;
}

/** Colours handed to collaborators, least-used first. Distinct enough to tell cursors apart. */
const USER_COLORS = [
    '#e6194b',
    '#3cb44b',
    '#4363d8',
    '#f58231',
    '#911eb4',
    '#008080',
    '#f032e6',
    '#9a6324',
    '#808000',
    '#000075',
];

const LOG_CAP = 2000;
/** Erased items are dropped from memory (and the saved file) after this many erasures, unless an undo still needs them. */
const COMPACT_AFTER_ERASURES = 1000;
const OP_ID_CAP = 20_000;
const UNDO_CAP = 200;
const MAX_LIVE_STROKES = 4;
const LEFTOVER_COLOR = '#555555';

export interface RoomOptions {
    onChange?: () => void;
    now?: () => number;
}

/**
 * One whiteboard. All state changes go through a single sequence number, so every client that
 * applies the log in order ends up with the same board. It knows nothing about sockets.
 */
export class Room {
    readonly board = new Board();
    seq = 0;
    private log: LogEntry[] = [];
    /** opId -> seq of the op it produced (or the seq at the time it turned out to be a no-op). */
    private readonly seen = new Map<string, number>();
    private readonly members = new Map<string, Member>();
    private readonly stacks = new Map<string, { undo: UndoEntry[]; redo: UndoEntry[] }>();
    private readonly colors = new Map<string, string>();
    private visibleCount = 0;
    private bytes = 0;
    private erasuresSinceCompact = 0;
    readonly id: string;
    private readonly onChange: () => void;
    private readonly now: () => number;

    constructor(id: string, options: RoomOptions = {}) {
        this.id = id;
        this.onChange = options.onChange ?? (() => {});
        this.now = options.now ?? Date.now;
    }

    static restore(data: Persisted, options?: RoomOptions): Room {
        const room = new Room(data.id, options);
        for (const [id, state] of Board.fromSnapshot(data.items).items) {
            room.board.items.set(id, state);
        }
        room.seq = data.seq;
        for (const s of room.board.items.values()) {
            room.bytes += approxBytes(s.item);
            if (!s.deleted) room.visibleCount++;
        }
        return room;
    }

    toPersisted(): Persisted {
        return { v: 1, id: this.id, seq: this.seq, items: this.board.snapshot() };
    }

    get memberCount(): number {
        return this.members.size;
    }

    join(conn: Conn, hello: Extract<ClientMsg, { t: 'hello' }>): void {
        const previous = this.members.get(hello.id);
        if (!previous && this.members.size >= MAX_MEMBERS_PER_ROOM) {
            conn.close(1013, 'room is full');
            return;
        }
        if (previous) {
            // Same tab reconnecting before the server noticed the old socket was dead.
            this.members.delete(hello.id);
            previous.conn.close(4001, 'replaced by a newer connection');
            this.broadcast({ t: 'leave', id: hello.id });
        }
        const user: User = {
            id: hello.id,
            name: cleanName(hello.name) || `Guest ${hello.id.slice(0, 4)}`,
            color: this.colorFor(hello.id),
        };
        const member: Member = {
            conn,
            user,
            ops: new Bucket(15, 60, this.now),
            ephemeral: new Bucket(120, 240, this.now),
            liveSids: new Map(),
        };
        const canCatchUp =
            hello.last !== null &&
            hello.last <= this.seq &&
            (hello.last === this.seq || (this.log[0] !== undefined && this.log[0].seq <= hello.last + 1));
        const stack = this.stacks.get(hello.id);
        const base = {
            t: 'welcome' as const,
            you: user,
            seq: this.seq,
            users: [...[...this.members.values()].map((m) => m.user), user],
            canUndo: (stack?.undo.length ?? 0) > 0,
            canRedo: (stack?.redo.length ?? 0) > 0,
        };
        if (canCatchUp && hello.last !== null) {
            const last = hello.last;
            conn.send({ ...base, mode: 'catchup', ops: this.log.filter((e) => e.seq > last) });
        } else {
            conn.send({ ...base, mode: 'snapshot', items: this.board.snapshot() });
        }
        this.members.set(hello.id, member);
        this.broadcast({ t: 'join', user }, user.id);
    }

    leave(id: string, conn: Conn): void {
        const member = this.members.get(id);
        if (!member || member.conn !== conn) return; // already replaced
        this.members.delete(id);
        for (const sid of member.liveSids.keys()) this.broadcast({ t: 'liveEnd', by: id, sid });
        this.broadcast({ t: 'leave', id });
    }

    receive(id: string, conn: Conn, msg: ClientMsg): void {
        const member = this.members.get(id);
        if (!member || member.conn !== conn) return;
        switch (msg.t) {
            case 'op':
                return this.handleOp(member, msg.opId, msg.op);
            case 'undo':
            case 'redo':
                return this.handleUndo(member, msg.opId, msg.t);
            case 'live': {
                if (!member.ephemeral.take()) return;
                const used = member.liveSids.get(msg.sid);
                if (used === undefined && member.liveSids.size >= MAX_LIVE_STROKES) return;
                const total = (used ?? 0) + msg.pts.length / 2;
                if (total > MAX_STROKE_POINTS) return; // an honest client commits and splits a stroke this long
                member.liveSids.set(msg.sid, total);
                return this.broadcast({ ...msg, by: id }, id);
            }
            case 'liveEnd':
                if (!member.ephemeral.take()) return;
                member.liveSids.delete(msg.sid);
                return this.broadcast({ t: 'liveEnd', by: id, sid: msg.sid }, id);
            case 'cur':
                if (!member.ephemeral.take()) return;
                return this.broadcast({ t: 'cur', by: id, x: msg.x, y: msg.y }, id);
            case 'name': {
                const name = cleanName(msg.name);
                if (!name || name === member.user.name || !member.ephemeral.take()) return;
                member.user = { ...member.user, name };
                return this.broadcast({ t: 'rename', id, name });
            }
            case 'ping':
                if (member.ephemeral.take()) member.conn.send({ t: 'pong' });
                return;
            case 'hello':
                return; // a second hello on a live connection is ignored
        }
    }

    private handleOp(member: Member, opId: string, requested: ClientOp): void {
        const known = this.seen.get(opId);
        if (known !== undefined) return member.conn.send({ t: 'ack', opId, seq: known });
        if (!member.ops.take()) return member.conn.send({ t: 'nack', opId, reason: 'rate' });

        let op: Op = requested;
        let entry: UndoEntry | null = null;
        switch (requested.k) {
            case 'add': {
                if (this.isFull(requested.item)) {
                    this.compact(); // erased items may be what is using the room up
                    if (this.isFull(requested.item)) return member.conn.send({ t: 'nack', opId, reason: 'full' });
                }
                const check = this.board.check(requested);
                if (check !== 'ok') return member.conn.send({ t: 'nack', opId, reason: check });
                const item = { ...requested.item, author: member.user.id };
                op = { k: 'add', item };
                entry = { undo: { k: 'del', ids: [item.id] }, redo: { k: 'res', ids: [item.id] } };
                break;
            }
            case 'del': {
                const ids = this.effectiveIds(requested.ids, (s) => !s.deleted);
                if (ids.length === 0) return this.noop(member, opId);
                op = { k: 'del', ids };
                entry = { undo: { k: 'res', ids }, redo: { k: 'del', ids } };
                break;
            }
            case 'mv': {
                const ids = this.effectiveIds(requested.ids, () => true);
                if (ids.length === 0) return member.conn.send({ t: 'nack', opId, reason: 'unknown' });
                // Moves add up, so many of them could push an offset past what a saved file can hold.
                const outOfRange = ids.some((id) => {
                    const s = this.board.items.get(id)!;
                    return Math.abs(s.dx + requested.dx) > MAX_OFFSET || Math.abs(s.dy + requested.dy) > MAX_OFFSET;
                });
                if (outOfRange) return member.conn.send({ t: 'nack', opId, reason: 'range' });
                op = { k: 'mv', ids, dx: requested.dx, dy: requested.dy };
                entry = {
                    undo: { k: 'mv', ids, dx: -requested.dx, dy: -requested.dy },
                    redo: op,
                };
                break;
            }
            case 'edit': {
                if (this.board.check(requested) !== 'ok') {
                    return member.conn.send({ t: 'nack', opId, reason: 'unknown' });
                }
                const state = this.board.items.get(requested.id);
                const previous = state?.item.t === 'text' ? state.item.text : '';
                if (previous === requested.text) return this.noop(member, opId);
                entry = {
                    undo: { k: 'edit', id: requested.id, text: previous },
                    redo: { k: 'edit', id: requested.id, text: requested.text },
                };
                break;
            }
        }
        this.commit(member.user.id, opId, op);
        if (op.k === 'add') member.liveSids.delete(op.item.id); // the committed stroke replaces its live copy
        const stack = this.stackOf(member.user.id);
        stack.undo.push(entry);
        if (stack.undo.length > UNDO_CAP) stack.undo.shift();
        stack.redo.length = 0; // a new action ends the redo branch
        this.sendUndoState(member);
    }

    private handleUndo(member: Member, opId: string, dir: 'undo' | 'redo'): void {
        const known = this.seen.get(opId);
        if (known !== undefined) return member.conn.send({ t: 'ack', opId, seq: known });
        if (!member.ops.take()) return member.conn.send({ t: 'nack', opId, reason: 'rate' });

        const stack = this.stackOf(member.user.id);
        const from = dir === 'undo' ? stack.undo : stack.redo;
        const to = dir === 'undo' ? stack.redo : stack.undo;
        // Skip entries whose effect has already happened (for example the stroke was erased by
        // someone else): the user pressed undo to see something change.
        for (let entry = from.pop(); entry; entry = from.pop()) {
            const op = dir === 'undo' ? entry.undo : entry.redo;
            if (!this.changesAnything(op)) continue;
            to.push(entry);
            this.commit(member.user.id, opId, op);
            return this.sendUndoState(member);
        }
        this.remember(opId, this.seq);
        member.conn.send({ t: 'nack', opId, reason: 'empty' });
        this.sendUndoState(member);
    }

    private changesAnything(op: Op): boolean {
        switch (op.k) {
            case 'del':
                return this.effectiveIds(op.ids, (s) => !s.deleted).length > 0;
            case 'res':
                return this.effectiveIds(op.ids, (s) => s.deleted).length > 0;
            case 'mv':
                return this.effectiveIds(op.ids, () => true).length > 0;
            case 'edit': {
                const state = this.board.items.get(op.id);
                return state?.item.t === 'text' && state.item.text !== op.text;
            }
            case 'add':
                return false;
        }
    }

    private effectiveIds(ids: string[], test: (s: NonNullable<ReturnType<Board['items']['get']>>) => boolean) {
        return ids.filter((id) => {
            const state = this.board.items.get(id);
            return state !== undefined && test(state);
        });
    }

    private noop(member: Member, opId: string): void {
        this.remember(opId, this.seq);
        member.conn.send({ t: 'ack', opId, seq: this.seq });
    }

    private isFull(item: Item): boolean {
        return this.visibleCount >= MAX_ITEMS_PER_ROOM || this.bytes + approxBytes(item) > MAX_ROOM_BYTES;
    }

    /**
     * Drop erased items that no undo or redo can bring back, so erasing really frees space (and
     * erased content stops being sent to newcomers and saved). Connected tabs that still hold the
     * tombstone are unaffected: an op that names a dropped id is simply skipped.
     */
    private compact(): void {
        const needed = new Set<string>();
        for (const s of this.stacks.values()) {
            for (const entry of [...s.undo, ...s.redo]) {
                for (const op of [entry.undo, entry.redo]) {
                    if ('ids' in op) for (const id of op.ids) needed.add(id);
                    else if (op.k === 'edit') needed.add(op.id);
                }
            }
        }
        let bytes = 0;
        for (const [id, state] of this.board.items) {
            if (state.deleted && !needed.has(id)) this.board.items.delete(id);
            else bytes += approxBytes(state.item);
        }
        this.bytes = bytes;
        this.erasuresSinceCompact = 0;
    }

    private commit(by: string, opId: string, op: Op): void {
        const entry: LogEntry = { seq: ++this.seq, by, opId, op };
        if (op.k === 'add') {
            this.visibleCount++;
            this.bytes += approxBytes(op.item);
        } else if (op.k === 'del' || op.k === 'res') {
            const erasing = op.k === 'del';
            for (const id of op.ids) {
                const state = this.board.items.get(id);
                if (state && state.deleted !== erasing) this.visibleCount += erasing ? -1 : 1;
            }
            if (erasing) this.erasuresSinceCompact += op.ids.length;
        }
        this.board.apply(op);
        if (this.erasuresSinceCompact >= COMPACT_AFTER_ERASURES) this.compact();
        this.log.push(entry);
        if (this.log.length > LOG_CAP) this.log.splice(0, this.log.length - LOG_CAP);
        this.remember(opId, entry.seq);
        this.broadcast({ t: 'op', ...entry });
        this.onChange();
    }

    private remember(opId: string, seq: number): void {
        this.seen.set(opId, seq);
        if (this.seen.size > OP_ID_CAP) {
            // Maps iterate in insertion order, so this drops the oldest.
            const oldest = this.seen.keys().next().value;
            if (oldest !== undefined) this.seen.delete(oldest);
        }
    }

    private sendUndoState(member: Member): void {
        const stack = this.stackOf(member.user.id);
        member.conn.send({
            t: 'undoState',
            canUndo: stack.undo.length > 0,
            canRedo: stack.redo.length > 0,
        });
    }

    private stackOf(id: string) {
        let stack = this.stacks.get(id);
        if (!stack) this.stacks.set(id, (stack = { undo: [], redo: [] }));
        return stack;
    }

    private colorFor(id: string): string {
        const remembered = this.colors.get(id);
        if (remembered) return remembered;
        const used = new Map<string, number>(USER_COLORS.map((c) => [c, 0]));
        for (const m of this.members.values()) used.set(m.user.color, (used.get(m.user.color) ?? 0) + 1);
        const color = [...used.entries()].sort((a, b) => a[1] - b[1])[0]?.[0] ?? LEFTOVER_COLOR;
        this.colors.set(id, color);
        if (this.colors.size > 1000) this.colors.delete(this.colors.keys().next().value as string);
        return color;
    }

    private broadcast(msg: ServerMsg, except?: string): void {
        // Serialize once, not once per collaborator: this is the hot path with many people in a room.
        let json: string | undefined;
        for (const [id, m] of this.members) {
            if (id === except) continue;
            json ??= JSON.stringify(msg);
            m.conn.send(msg, json);
        }
    }
}

/** A rough in-memory size for budgeting, cheaper than serializing the item. */
export function approxBytes(item: Item): number {
    switch (item.t) {
        case 'stroke':
            return 120 + item.pts.length * 8 + (item.pr?.length ?? 0) * 5;
        case 'text':
            return 160 + item.text.length * 2;
        case 'shape':
            return 160;
    }
}

/**
 * Display names are text only. Drop control characters, zero-width and bidirectional-override
 * characters (they let a name pretend to be another, or reorder what is next to it), and cap the length.
 */
export function cleanName(raw: string): string {
    return raw
        .replace(/[\u0000-\u001f\u007f-\u009f​-‏‪-‮⁦-⁩﻿]/g, '')
        .trim()
        .slice(0, MAX_NAME);
}
