/**
 * `localStorage` that can never throw: blocked storage (private modes, disabled cookies, a sandboxed
 * frame) makes the getter itself throw, and a full quota makes `setItem` throw. Both fall back to
 * memory for the life of the page, so the feature still works, it just forgets on reload.
 */
export function safeStorage(getStore = () => window.localStorage) {
    const memory = new Map();
    let store = null;
    try {
        store = getStore();
    } catch {
        // blocked: stay in memory
    }
    return {
        get(key) {
            try {
                const value = store?.getItem(key);
                if (value !== undefined && value !== null) return value;
            } catch {
                // fall through to memory
            }
            return memory.get(key) ?? null;
        },
        set(key, value) {
            memory.set(key, value);
            try {
                store?.setItem(key, value);
            } catch {
                // quota or blocked: memory keeps it
            }
        },
        remove(key) {
            memory.delete(key);
            try {
                store?.removeItem(key);
            } catch {
                // nothing to do
            }
        },
    };
}
