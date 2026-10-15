import type { Item, Op, SnapshotItem } from './types.ts';

export interface ItemState {
    item: Item;
    /** Erased items stay in the map as tombstones so a later `res` (undo) can bring them back. */
    deleted: boolean;
    /** Accumulated moves. Moves add up, so two people dragging the same item never overwrite each other. */
    dx: number;
    dy: number;
}

export type Check = 'ok' | 'dup' | 'unknown';

/**
 * The confirmed whiteboard: items in z-order (insertion order of their `add`) plus tombstones.
 * The server and every client run the same `apply` over the same sequenced ops, so they converge.
 */
export class Board {
    readonly items = new Map<string, ItemState>();

    /** Would `op` change anything real? `dup`: an `add` with a taken id. `unknown`: it names nothing we have. */
    check(op: Op): Check {
        switch (op.k) {
            case 'add':
                return this.items.has(op.item.id) ? 'dup' : 'ok';
            case 'del':
            case 'res':
            case 'mv':
                return op.ids.some((id) => this.items.has(id)) ? 'ok' : 'unknown';
            case 'edit':
                return this.items.get(op.id)?.item.t === 'text' ? 'ok' : 'unknown';
        }
    }

    /** Apply a sequenced op. Ids that do not exist are skipped, so applying is always safe. */
    apply(op: Op): void {
        switch (op.k) {
            case 'add':
                if (!this.items.has(op.item.id)) {
                    this.items.set(op.item.id, { item: op.item, deleted: false, dx: 0, dy: 0 });
                }
                return;
            case 'del':
            case 'res':
                for (const id of op.ids) {
                    const state = this.items.get(id);
                    if (state) state.deleted = op.k === 'del';
                }
                return;
            case 'mv':
                for (const id of op.ids) {
                    const state = this.items.get(id);
                    if (state) {
                        state.dx += op.dx;
                        state.dy += op.dy;
                    }
                }
                return;
            case 'edit': {
                const state = this.items.get(op.id);
                if (state && state.item.t === 'text') state.item = { ...state.item, text: op.text };
                return;
            }
        }
    }

    /** A deep copy that is safe to hand to another board or to serialize. */
    snapshot(): SnapshotItem[] {
        return [...this.items.values()].map((s) => structuredClone(s));
    }

    static fromSnapshot(items: SnapshotItem[]): Board {
        const board = new Board();
        for (const s of items) board.items.set(s.item.id, structuredClone(s));
        return board;
    }

    /** The visible items only, in z-order. */
    visible(): ItemState[] {
        return [...this.items.values()].filter((s) => !s.deleted);
    }
}
