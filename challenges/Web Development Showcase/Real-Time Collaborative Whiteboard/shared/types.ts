/** Wire and state types shared by the server and the browser client. No runtime code here. */

export interface ItemBase {
    id: string;
    /** Set by the server from the connection, never trusted from the client. */
    author: string;
    /** `#rrggbb`. */
    color: string;
    /** Stroke width (or font size basis) in world units. */
    size: number;
}

export interface StrokeItem extends ItemBase {
    t: 'stroke';
    /** Flat `x0, y0, x1, y1, ...` in world units. */
    pts: number[];
    /** Optional per-point pressure (0..1), present for pens only. */
    pr?: number[];
    /** Highlighter: wide and translucent. */
    hl?: boolean;
}

export type ShapeKind = 'line' | 'arrow' | 'rect' | 'ellipse';

export interface ShapeItem extends ItemBase {
    t: 'shape';
    k: ShapeKind;
    x1: number;
    y1: number;
    x2: number;
    y2: number;
}

export interface TextItem extends ItemBase {
    t: 'text';
    x: number;
    y: number;
    text: string;
    /** Sticky note: `color` is the paper colour and the text is dark. */
    sticky: boolean;
}

export type Item = StrokeItem | ShapeItem | TextItem;

/**
 * What a client asks for. The server sequences these, so every client applies the same ops in the
 * same order. Undo and redo are not ops: the server turns them into concrete `del`/`res`/`mv`/`edit`.
 */
export type Op =
    | { k: 'add'; item: Item }
    | { k: 'del'; ids: string[] }
    | { k: 'res'; ids: string[] }
    | { k: 'mv'; ids: string[]; dx: number; dy: number }
    | { k: 'edit'; id: string; text: string };

/** `res` only ever comes from the server (as the result of an undo), never from a client. */
export type ClientOp = Exclude<Op, { k: 'res' }>;

export interface User {
    id: string;
    name: string;
    color: string;
}

export interface LiveStart {
    color: string;
    size: number;
    hl?: boolean;
}

export type ClientMsg =
    | { t: 'hello'; v: 1; id: string; name: string; last: number | null }
    | { t: 'op'; opId: string; op: ClientOp }
    | { t: 'undo'; opId: string }
    | { t: 'redo'; opId: string }
    | { t: 'live'; sid: string; pts: number[]; pr?: number[]; start?: LiveStart }
    | { t: 'liveEnd'; sid: string }
    | { t: 'cur'; x: number; y: number }
    | { t: 'name'; name: string }
    | { t: 'ping' };

export interface LogEntry {
    seq: number;
    by: string;
    /** The client's id for the request, used to match its own ops and to deduplicate resends. */
    opId: string;
    op: Op;
}

export interface SnapshotItem {
    item: Item;
    deleted: boolean;
    dx: number;
    dy: number;
}

export type ServerMsg =
    | {
        t: 'welcome';
        you: User;
        seq: number;
        /** `snapshot` replaces the client's state; `catchup` is the ops after its `last`. */
        mode: 'snapshot' | 'catchup';
        items?: SnapshotItem[];
        ops?: LogEntry[];
        users: User[];
        canUndo: boolean;
        canRedo: boolean;
    }
    | ({ t: 'op' } & LogEntry)
    | { t: 'ack'; opId: string; seq: number }
    | { t: 'nack'; opId: string; reason: string }
    | { t: 'undoState'; canUndo: boolean; canRedo: boolean }
    | { t: 'live'; by: string; sid: string; pts: number[]; pr?: number[]; start?: LiveStart }
    | { t: 'liveEnd'; by: string; sid: string }
    | { t: 'cur'; by: string; x: number; y: number }
    | { t: 'join'; user: User }
    | { t: 'leave'; id: string }
    | { t: 'rename'; id: string; name: string }
    | { t: 'pong' };
