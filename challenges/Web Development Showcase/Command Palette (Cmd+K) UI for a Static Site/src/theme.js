import { safeStorage } from './storage.js';

export const THEME_KEY = 'palette:theme';
const ORDER = ['system', 'light', 'dark'];

/** `system` is no override: the stylesheet follows `prefers-color-scheme`. */
export function createTheme(root = document.documentElement, storage = safeStorage()) {
    function current() {
        const stored = storage.get(THEME_KEY);
        return ORDER.includes(stored) ? stored : 'system';
    }
    function apply(theme) {
        if (theme === 'system') delete root.dataset.theme;
        else root.dataset.theme = theme;
    }
    apply(current());
    return {
        current,
        /** system -> light -> dark -> system. Returns the new theme. */
        cycle() {
            const next = ORDER[(ORDER.indexOf(current()) + 1) % ORDER.length];
            if (next === 'system') storage.remove(THEME_KEY);
            else storage.set(THEME_KEY, next);
            apply(next);
            return next;
        },
    };
}
