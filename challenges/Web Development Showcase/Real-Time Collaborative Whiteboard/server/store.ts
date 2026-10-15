import { createHash } from 'node:crypto';
import { mkdirSync, readdirSync, readFileSync, renameSync } from 'node:fs';
import { rename, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { z } from 'zod';
import { MAX_COORD } from '../shared/limits.ts';
import type { Item } from '../shared/types.ts';
import type { Persisted } from './room.ts';
import { persistedItem } from './schema.ts';

const coord = z.number().min(-MAX_COORD * 100).max(MAX_COORD * 100);

const file = z.object({
    v: z.literal(1),
    id: z.string(),
    seq: z.number().int().min(0),
    items: z.array(
        z.object({
            item: z.unknown(),
            deleted: z.boolean(),
            dx: coord,
            dy: coord,
        }),
    ),
});

/**
 * One JSON file per room. The file name is a hash of the room id, so a hostile id can never
 * escape the data directory or hit a reserved Windows name like `CON`. Writes are atomic
 * (temp file, then rename) so a crash leaves the previous version, never half a file.
 */
export class RoomStore {
    private readonly pending = new Map<string, Promise<void>>();
    private readonly dir: string;

    constructor(dir: string) {
        this.dir = dir;
        mkdirSync(dir, { recursive: true });
    }

    private path(id: string): string {
        return join(this.dir, `${createHash('sha256').update(id).digest('hex').slice(0, 32)}.json`);
    }

    /** Number of saved rooms, used to cap how many an anonymous visitor can create. */
    count(): number {
        return readdirSync(this.dir).filter((f) => f.endsWith('.json')).length;
    }

    /** Load and validate. A missing file is `null`; a corrupt one is moved aside and treated as new. */
    load(id: string): Persisted | null {
        const path = this.path(id);
        let text: string;
        try {
            text = readFileSync(path, 'utf8');
        } catch {
            return null;
        }
        try {
            const parsed = file.parse(JSON.parse(text));
            if (parsed.id !== id) throw new Error('room id mismatch');
            const items = parsed.items.map((s) => {
                const raw = s.item as { author?: unknown };
                const item = persistedItem.parse(raw);
                const author = typeof raw.author === 'string' ? raw.author.slice(0, 64) : 'unknown';
                return { ...s, item: { ...item, author } as Item };
            });
            return { v: 1, id, seq: parsed.seq, items };
        } catch {
            try {
                renameSync(path, `${path}.corrupt-${Date.now()}`);
            } catch {
                // nothing more to do: the room starts empty and the next save replaces the file
            }
            return null;
        }
    }

    /** Saves are serialized per room so an older snapshot can never land after a newer one. */
    save(id: string, data: Persisted): Promise<void> {
        const path = this.path(id);
        const tmp = `${path}.${process.pid}.tmp`;
        const body = JSON.stringify(data);
        const previous = this.pending.get(id) ?? Promise.resolve();
        const next = previous
            .catch(() => {})
            .then(async () => {
                await writeFile(tmp, body);
                await rename(tmp, path);
            });
        this.pending.set(id, next);
        return next.finally(() => {
            if (this.pending.get(id) === next) this.pending.delete(id);
        });
    }
}
