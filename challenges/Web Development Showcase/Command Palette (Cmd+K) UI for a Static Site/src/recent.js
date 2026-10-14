import { safeStorage } from './storage.js';

export const RECENT_KEY = 'palette:recent';
export const RECENT_VERSION = 1;
export const RECENT_MAX = 5;
const MAX_ID_LENGTH = 200;

/**
 * Recently chosen item ids, newest first. What comes back from storage is never trusted: a wrong
 * version, a non-array, non-string or oversized ids, duplicates and anything past the cap are all
 * dropped, and unreadable JSON is treated as empty (and overwritten on the next save).
 */
export function createRecents(storage = safeStorage()) {
    function read() {
        let parsed;
        try {
            parsed = JSON.parse(storage.get(RECENT_KEY) ?? 'null');
        } catch {
            return [];
        }
        if (!parsed || parsed.v !== RECENT_VERSION || !Array.isArray(parsed.ids)) return [];
        const ids = [];
        for (const id of parsed.ids) {
            if (typeof id === 'string' && id && id.length <= MAX_ID_LENGTH && !ids.includes(id)) ids.push(id);
        }
        return ids.slice(0, RECENT_MAX);
    }

    function write(ids) {
        storage.set(RECENT_KEY, JSON.stringify({ v: RECENT_VERSION, ids }));
    }

    return {
        get: read,
        push(id) {
            write([id, ...read().filter((existing) => existing !== id)].slice(0, RECENT_MAX));
        },
        clear() {
            storage.remove(RECENT_KEY);
        },
    };
}
