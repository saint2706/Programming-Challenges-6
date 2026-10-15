import { Room } from './room.ts';
import type { RoomStore } from './store.ts';

export interface RoomsOptions {
    /** Rooms kept in memory at once. */
    maxLoaded?: number;
    /** Rooms kept on disk. Stops an anonymous visitor from filling the disk with new boards. */
    maxSaved?: number;
    /** Wait this long after the last change before writing (one write per burst of drawing). */
    saveDelayMs?: number;
    /** An empty room is flushed and dropped from memory after this long. */
    idleMs?: number;
}

export type Opened = { room: Room } | { error: 'busy' | 'full' };

interface Slot {
    room: Room;
    saveTimer?: NodeJS.Timeout;
    idleTimer?: NodeJS.Timeout;
}

/** Loads rooms on demand, saves them after changes, and unloads them when everyone has left. */
export class Rooms {
    private readonly slots = new Map<string, Slot>();
    private readonly maxLoaded: number;
    private readonly maxSaved: number;
    private readonly saveDelayMs: number;
    private readonly idleMs: number;
    private readonly store: RoomStore;

    constructor(store: RoomStore, options: RoomsOptions = {}) {
        this.store = store;
        this.maxLoaded = options.maxLoaded ?? 200;
        this.maxSaved = options.maxSaved ?? 2000;
        this.saveDelayMs = options.saveDelayMs ?? 400;
        this.idleMs = options.idleMs ?? 30_000;
    }

    get loaded(): number {
        return this.slots.size;
    }

    open(id: string): Opened {
        const existing = this.slots.get(id);
        if (existing) {
            clearTimeout(existing.idleTimer);
            return { room: existing.room };
        }
        if (this.slots.size >= this.maxLoaded) return { error: 'busy' };
        const saved = this.store.load(id);
        if (!saved && this.store.count() >= this.maxSaved) return { error: 'full' };
        const options = { onChange: () => this.scheduleSave(id) };
        const room = saved ? Room.restore(saved, options) : new Room(id, options);
        this.slots.set(id, { room });
        return { room };
    }

    /** Call when a connection leaves; unloads the room once nobody is in it. */
    release(id: string): void {
        const slot = this.slots.get(id);
        if (!slot || slot.room.memberCount > 0) return;
        clearTimeout(slot.idleTimer);
        slot.idleTimer = setTimeout(() => {
            if (slot.room.memberCount > 0) return;
            void this.flush(id).then(() => {
                if (this.slots.get(id) === slot && slot.room.memberCount === 0) this.slots.delete(id);
            });
        }, this.idleMs);
        slot.idleTimer.unref();
    }

    private scheduleSave(id: string): void {
        const slot = this.slots.get(id);
        if (!slot || slot.saveTimer) return;
        slot.saveTimer = setTimeout(() => {
            slot.saveTimer = undefined;
            void this.write(slot);
        }, this.saveDelayMs);
        slot.saveTimer.unref();
    }

    private async write(slot: Slot): Promise<void> {
        try {
            await this.store.save(slot.room.id, slot.room.toPersisted());
        } catch (error) {
            // A failed save must not take the server down; the next change retries.
            console.error(`could not save room ${slot.room.id}:`, error);
        }
    }

    private async flush(id: string): Promise<void> {
        const slot = this.slots.get(id);
        if (!slot) return;
        clearTimeout(slot.saveTimer);
        slot.saveTimer = undefined;
        if (slot.room.seq > 0) await this.write(slot);
    }

    /** Write every room now (shutdown). */
    async flushAll(): Promise<void> {
        await Promise.all([...this.slots.keys()].map((id) => this.flush(id)));
    }

    dispose(): void {
        for (const slot of this.slots.values()) {
            clearTimeout(slot.saveTimer);
            clearTimeout(slot.idleTimer);
        }
    }
}
