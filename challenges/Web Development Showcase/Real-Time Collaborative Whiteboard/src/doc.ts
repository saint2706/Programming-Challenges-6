import { Board, type ItemState } from '../shared/board.ts';
import { MAX_STROKE_POINTS } from '../shared/limits.ts';
import type { ClientMsg, ClientOp, Item, LiveStart, ServerMsg, User } from '../shared/types.ts';

interface Pending {
    opId: string;
    msg: Extract<ClientMsg, { t: 'op' | 'undo' | 'redo' }>;
}

export interface LiveStroke {
    by: string;
    sid: string;
    start: LiveStart;
    pts: number[];
    pr: number[] | null;
    /** `performance.now()` of the last chunk, so a stroke whose sender vanished can be dropped. */
    at: number;
}

export interface Cursor {
    x: number;
    y: number;
    at: number;
}

/** Your own ops that the server has not echoed yet, folded into one lookup per item. */
interface Layer {
    adds: Map<string, ItemState>;
    deleted: Set<string>;
    moves: Map<string, { dx: number; dy: number }>;
    edits: Map<string, string>;
}

export interface Visible {
    item: Item;
    dx: number;
    dy: number;
}

/**
 * What one browser tab knows about the board: the server-confirmed state plus your own ops that are
 * still in flight, shown optimistically. Confirmed ops arrive in server order, so every client's
 * confirmed state is identical; the pending layer only ever holds your own not-yet-echoed ops.
 */
export class Doc {
    confirmed = new Board();
    seq = 0;
    /** Bumped on every change at all (cursors, users, live strokes included). */
    version = 0;
    /** Bumped only when the committed items on screen may have changed. */
    boardVersion = 0;
    synced = false;
    me: User | null = null;
    users = new Map<string, User>();
    canUndo = false;
    canRedo = false;
    readonly live = new Map<string, LiveStroke>();
    readonly cursors = new Map<string, Cursor>();
    private pending: Pending[] = [];
    private layer: Layer = emptyLayer();
    onChange: () => void = () => {};
    onNack: (reason: string) => void = () => {};
    private readonly send: (msg: ClientMsg) => void;
    private readonly newId: () => string;
    private readonly clientId: string;
    private readonly now: () => number;

    constructor(
        send: (msg: ClientMsg) => void,
        newId: () => string,
        clientId: string,
        now: () => number = () => performance.now(),
    ) {
        this.send = send;
        this.newId = newId;
        this.clientId = clientId;
        this.now = now;
    }

    /** Number of your ops the server has not confirmed yet. */
    get pendingCount(): number {
        return this.pending.length;
    }

    hello(name: string): ClientMsg {
        return { t: 'hello', v: 1, id: this.clientId, name, last: this.synced || this.seq > 0 ? this.seq : null };
    }

    /** Apply an op locally right away and send it. Returns its id. */
    submit(op: ClientOp): string {
        const opId = this.newId();
        const msg = { t: 'op', opId, op } as const;
        this.pending.push({ opId, msg });
        this.rebuildLayer();
        this.send(msg);
        this.boardVersion++;
        this.changed();
        return opId;
    }

    /** Undo and redo are decided by the server (it keeps your history), so nothing is applied here. */
    undo(): void {
        this.request('undo');
    }

    redo(): void {
        this.request('redo');
    }

    private request(t: 'undo' | 'redo'): void {
        const opId = this.newId();
        const msg = { t, opId } as const;
        this.pending.push({ opId, msg });
        this.send(msg);
    }

    /** Connection lost: keep everything, forget who is around and any half-seen strokes. */
    disconnected(): void {
        this.synced = false;
        this.live.clear();
        this.cursors.clear();
        this.users.clear();
        this.changed();
    }

    /** Re-send what the server may not have seen. It deduplicates by opId, so over-sending is safe. */
    resendPending(): void {
        for (const p of this.pending) this.send(p.msg);
    }

    handle(msg: ServerMsg): void {
        switch (msg.t) {
            case 'welcome':
                this.me = msg.you;
                this.users = new Map(msg.users.map((u) => [u.id, u]));
                this.canUndo = msg.canUndo;
                this.canRedo = msg.canRedo;
                if (msg.mode === 'snapshot') {
                    this.confirmed = Board.fromSnapshot(msg.items ?? []);
                } else {
                    for (const entry of msg.ops ?? []) this.applyConfirmed(entry.op, entry.opId);
                }
                this.seq = msg.seq;
                this.synced = true;
                this.rebuildLayer();
                this.resendPending();
                this.boardVersion++;
                return this.changed();
            case 'op':
                if (msg.seq <= this.seq) return; // a repeat from a catch-up overlap
                this.seq = msg.seq;
                this.applyConfirmed(msg.op, msg.opId);
                if (msg.op.k === 'add') this.live.delete(`${msg.by}:${msg.op.item.id}`);
                this.rebuildLayer();
                this.boardVersion++;
                return this.changed();
            case 'ack':
                this.drop(msg.opId);
                this.boardVersion++;
                return this.changed();
            case 'nack':
                this.drop(msg.opId);
                this.onNack(msg.reason);
                this.boardVersion++;
                return this.changed();
            case 'undoState':
                this.canUndo = msg.canUndo;
                this.canRedo = msg.canRedo;
                return this.changed();
            case 'live': {
                const key = `${msg.by}:${msg.sid}`;
                let stroke = this.live.get(key);
                if (!stroke) {
                    if (!msg.start) return; // the first chunk was lost; wait for the committed stroke
                    stroke = { by: msg.by, sid: msg.sid, start: msg.start, pts: [], pr: msg.pr ? [] : null, at: 0 };
                    this.live.set(key, stroke);
                }
                // The server caps this too; a well-behaved sender commits and splits long before.
                if (stroke.pts.length / 2 + msg.pts.length / 2 > MAX_STROKE_POINTS) return;
                stroke.pts.push(...msg.pts);
                if (stroke.pr && msg.pr) stroke.pr.push(...msg.pr);
                stroke.at = this.now();
                return this.changed();
            }
            case 'liveEnd':
                this.live.delete(`${msg.by}:${msg.sid}`);
                return this.changed();
            case 'cur':
                this.cursors.set(msg.by, { x: msg.x, y: msg.y, at: this.now() });
                return this.changed();
            case 'join':
                this.users.set(msg.user.id, msg.user);
                return this.changed();
            case 'leave':
                this.users.delete(msg.id);
                this.cursors.delete(msg.id);
                return this.changed();
            case 'rename': {
                const user = this.users.get(msg.id);
                if (user) this.users.set(msg.id, { ...user, name: msg.name });
                if (this.me?.id === msg.id) this.me = { ...this.me, name: msg.name };
                return this.changed();
            }
            case 'pong':
                return;
        }
    }

    /** Visible items in z-order with their current offsets, confirmed first, then your pending adds. */
    forEachVisible(fn: (v: Visible) => void): void {
        for (const state of this.confirmed.items.values()) this.emit(state, fn);
        for (const state of this.layer.adds.values()) this.emit(state, fn);
    }

    /** One visible item by id (confirmed or still pending), or `undefined` if it is gone or erased. */
    find(id: string): Visible | undefined {
        const state = this.confirmed.items.get(id) ?? this.layer.adds.get(id);
        let found: Visible | undefined;
        if (state) this.emit(state, (v) => (found = v));
        return found;
    }

    visible(): Visible[] {
        const out: Visible[] = [];
        this.forEachVisible((v) => out.push(v));
        return out;
    }

    private emit(state: ItemState, fn: (v: Visible) => void): void {
        const id = state.item.id;
        if (state.deleted || this.layer.deleted.has(id)) return;
        const move = this.layer.moves.get(id);
        const text = this.layer.edits.get(id);
        const item = text !== undefined && state.item.t === 'text' ? { ...state.item, text } : state.item;
        fn({ item, dx: state.dx + (move?.dx ?? 0), dy: state.dy + (move?.dy ?? 0) });
    }

    private applyConfirmed(op: Parameters<Board['apply']>[0], opId: string): void {
        this.confirmed.apply(op);
        this.drop(opId);
    }

    private drop(opId: string): void {
        const before = this.pending.length;
        this.pending = this.pending.filter((p) => p.opId !== opId);
        if (this.pending.length !== before) this.rebuildLayer();
    }

    private rebuildLayer(): void {
        const layer = emptyLayer();
        const author = this.me?.id ?? '';
        for (const { msg } of this.pending) {
            if (msg.t !== 'op') continue;
            const op = msg.op;
            switch (op.k) {
                case 'add':
                    layer.adds.set(op.item.id, {
                        item: { ...op.item, author },
                        deleted: false,
                        dx: 0,
                        dy: 0,
                    });
                    break;
                case 'del':
                    for (const id of op.ids) layer.deleted.add(id);
                    break;
                case 'mv':
                    for (const id of op.ids) {
                        const m = layer.moves.get(id) ?? { dx: 0, dy: 0 };
                        layer.moves.set(id, { dx: m.dx + op.dx, dy: m.dy + op.dy });
                    }
                    break;
                case 'edit':
                    layer.edits.set(op.id, op.text);
                    break;
            }
        }
        // A pending add that the server already confirmed (its echo is in `confirmed`) is not shown twice.
        for (const id of layer.adds.keys()) if (this.confirmed.items.has(id)) layer.adds.delete(id);
        this.layer = layer;
    }

    private changed(): void {
        this.version++;
        this.onChange();
    }
}

function emptyLayer(): Layer {
    return { adds: new Map(), deleted: new Set(), moves: new Map(), edits: new Map() };
}
