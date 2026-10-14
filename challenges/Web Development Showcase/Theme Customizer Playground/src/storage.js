import { defaults, validate } from './schema.js';

export const KEY = 'theme-playground:v1';
export const BACKUP_KEY = 'theme-playground:backup';

/**
 * Autosave for the current theme. `storage` is a localStorage-like object, or null when the browser
 * has none. Whatever is in there is untrusted: it is parsed and validated on load, and a bad value is
 * moved aside (not deleted) so it can be inspected, and never half-applied.
 */
export function createStore(storage) {
  return {
    /** `{ theme, notice }`: a whole valid theme (the defaults if nothing usable is stored). */
    load() {
      if (!storage) return { theme: defaults(), notice: 'Saving is off in this browser, so changes will not be kept.' };
      let text;
      try {
        text = storage.getItem(KEY);
      } catch {
        return { theme: defaults(), notice: 'Saving is off in this browser, so changes will not be kept.' };
      }
      if (text === null) return { theme: defaults(), notice: null };
      let result = { ok: false };
      try {
        const saved = JSON.parse(text);
        if (saved && saved.v === 1) result = validate(saved.theme);
      } catch {
        // fall through: treated as corrupt
      }
      if (result.ok) return { theme: result.theme, notice: null };
      try {
        storage.setItem(BACKUP_KEY, text.slice(0, 20_000));
        storage.removeItem(KEY);
      } catch {
        // best effort
      }
      return { theme: defaults(), notice: 'The saved theme was unreadable, so it was set aside and the default theme loaded.' };
    },
    /** True when the theme was saved. */
    save(theme) {
      if (!storage) return false;
      try {
        storage.setItem(KEY, JSON.stringify({ v: 1, theme }));
        return true;
      } catch {
        return false;
      }
    },
  };
}
