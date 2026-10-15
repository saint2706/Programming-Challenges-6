import type { Board } from '../shared/board.ts';
import type { ClientMsg, Item, ServerMsg } from '../shared/types.ts';
import { Doc } from '../src/doc.ts';
import { Room, type Conn } from '../server/room.ts';

export const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value));

export function stroke(id: string, x = 0, y = 0): Item {
    return { t: 'stroke', id, author: 'x', color: '#112233', size: 4, pts: [x, y, x + 10, y + 5] };
}

export function textItem(id: string, text = 'hi'): Item {
    return { t: 'text', id, author: 'x', color: '#112233', size: 24, x: 5, y: 5, text, sticky: false };
}

/** A fake connection that records what the room sends, as the wire would carry it. */
export class Recorder implements Conn {
    readonly got: ServerMsg[] = [];
    closed: { code: number; reason: string } | null = null;
    send(msg: ServerMsg): void {
        this.got.push(clone(msg));
    }
    close(code: number, reason: string): void {
        this.closed = { code, reason };
    }
    ofType<T extends ServerMsg['t']>(t: T): Extract<ServerMsg, { t: T }>[] {
        return this.got.filter((m): m is Extract<ServerMsg, { t: T }> => m.t === t);
    }
}

export function hello(id: string, name = id, last: number | null = null): Extract<ClientMsg, { t: 'hello' }> {
    return { t: 'hello', v: 1, id, name, last };
}

/** Join a fake connection to a room and return it. */
export function join(room: Room, id: string, last: number | null = null): Recorder {
    const rec = new Recorder();
    room.join(rec, hello(id, id, last));
    return rec;
}

/** A canonical, comparable view of a board (order is z-order). */
export function describe(board: Board) {
    return [...board.items.values()].map((s) => [s.item.id, s.deleted, s.dx, s.dy, clone(s.item)]);
}

/**
 * A network of `n` browser tabs and one room where every message waits in a queue until a test
 * delivers it, so tests can reorder work between clients, lose messages and reconnect.
 */
export class Net {
    room: Room;
    readonly docs: Doc[] = [];
    private readonly up: ClientMsg[][] = [];
    private readonly down: ServerMsg[][] = [];
    private readonly conns: (Recorder | null)[] = [];
    private ids = 0;

    private clock = 0;
    /** Each call moves time on 100 ms, so the room's rate limits never trip in these tests. */
    private readonly now = () => (this.clock += 100);

    readonly n: number;

    constructor(n: number) {
        this.n = n;
        this.room = new Room('r', { now: this.now });
        for (let i = 0; i < n; i++) {
            this.up.push([]);
            this.down.push([]);
            this.conns.push(null);
            const clientId = `c${i}`;
            const doc = new Doc(
                (msg) => {
                    if (this.conns[i]) this.up[i]!.push(clone(msg));
                },
                () => `id${this.ids++}`,
                clientId,
                () => 0,
            );
            this.docs.push(doc);
        }
    }

    connected(i: number): boolean {
        return this.conns[i] !== null;
    }

    connect(i: number): void {
        if (this.conns[i]) return;
        const rec = new Recorder();
        // Everything the room sends this tab lands in its downstream queue.
        rec.send = (msg) => this.down[i]!.push(clone(msg));
        this.conns[i] = rec;
        this.up[i]!.push(this.docs[i]!.hello(`tab ${i}`));
    }

    /** Lose the connection: whatever was in flight either way is gone. */
    disconnect(i: number): void {
        const conn = this.conns[i];
        if (!conn) return;
        this.conns[i] = null;
        this.up[i]!.length = 0;
        this.down[i]!.length = 0;
        this.room.leave(`c${i}`, conn);
        this.docs[i]!.disconnected();
    }

    /** Server restart: rooms reload from their saved form and every tab loses its connection. */
    restart(): void {
        for (let i = 0; i < this.n; i++) this.disconnect(i);
        this.room = Room.restore(clone(this.room.toPersisted()), { now: this.now });
    }

    pendingUp(i: number): number {
        return this.up[i]!.length;
    }

    pendingDown(i: number): number {
        return this.down[i]!.length;
    }

    /** Deliver the oldest queued message from tab `i` to the room. */
    deliverUp(i: number): void {
        const msg = this.up[i]!.shift();
        const conn = this.conns[i];
        if (!msg || !conn) return;
        if (msg.t === 'hello') this.room.join(conn, msg);
        else this.room.receive(`c${i}`, conn, msg);
    }

    /** Deliver the oldest queued message from the room to tab `i`. */
    deliverDown(i: number): void {
        const msg = this.down[i]!.shift();
        if (msg) this.docs[i]!.handle(msg);
    }

    /** Reconnect everyone and deliver until nothing is left in flight. */
    settle(): void {
        for (let i = 0; i < this.n; i++) this.connect(i);
        this.flush();
    }

    /** Deliver everything queued for the tabs that are connected, without reconnecting anyone. */
    flush(): void {
        for (let guard = 0; guard < 100_000; guard++) {
            let moved = false;
            for (let i = 0; i < this.n; i++) {
                if (this.up[i]!.length) (this.deliverUp(i), (moved = true));
                if (this.down[i]!.length) (this.deliverDown(i), (moved = true));
            }
            if (!moved) return;
        }
        throw new Error('network did not settle');
    }
}
