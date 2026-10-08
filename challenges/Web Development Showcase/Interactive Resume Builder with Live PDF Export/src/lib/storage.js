import { parseResume } from './schema.js';

/**
 * Autosave to localStorage, written so that storage being unavailable, full or corrupt never
 * breaks the app. Every access is wrapped; a value that cannot be read back as a valid resume is
 * moved to a backup key (not deleted) and reported as `corrupt`.
 */

export const STORAGE_KEY = 'resume-builder:v1';
export const CORRUPT_KEY = 'resume-builder:v1:corrupt';
export const STORAGE_VERSION = 1;

/** `localStorage`, or null where touching it throws (blocked cookies, sandboxed frames). */
function defaultBackend() {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

export function createStorage(backend = defaultBackend()) {
  let usable = backend !== null && backend !== undefined;
  if (usable) {
    try {
      backend.getItem(STORAGE_KEY);
    } catch {
      usable = false;
    }
  }

  function corrupt(raw, reason) {
    try {
      backend.setItem(CORRUPT_KEY, raw);
    } catch {
      // nowhere to keep the backup; still report the problem
    }
    return { status: 'corrupt', reason };
  }

  return {
    available: usable,

    /** `{status: 'empty'}`, `{status: 'ok', doc}` or `{status: 'corrupt', reason}`. */
    load() {
      if (!usable) return { status: 'empty' };
      let raw;
      try {
        raw = backend.getItem(STORAGE_KEY);
      } catch {
        return { status: 'empty' };
      }
      if (raw === null) return { status: 'empty' };
      let saved;
      try {
        saved = JSON.parse(raw);
      } catch {
        return corrupt(raw, 'not valid JSON');
      }
      if (saved === null || typeof saved !== 'object' || saved.version !== STORAGE_VERSION) {
        return corrupt(raw, 'unknown format');
      }
      const result = parseResume(saved.doc);
      if (!result.ok) return corrupt(raw, 'invalid document');
      return { status: 'ok', doc: result.data };
    },

    /** True if the document was written. */
    save(doc) {
      if (!usable) return false;
      try {
        backend.setItem(
          STORAGE_KEY,
          JSON.stringify({ version: STORAGE_VERSION, savedAt: new Date().toISOString(), doc }),
        );
        return true;
      } catch {
        return false;
      }
    },

    clear() {
      if (!usable) return;
      try {
        backend.removeItem(STORAGE_KEY);
      } catch {
        // nothing to do
      }
    },
  };
}

/** Debounced saving: only the latest scheduled document is written, after `delay` ms of quiet. */
export function createAutosaver(save, delay = 400) {
  let timer = null;
  let latest;
  let dirty = false;

  const write = () => {
    timer = null;
    if (!dirty) return;
    dirty = false;
    save(latest);
  };

  return {
    schedule(doc) {
      latest = doc;
      dirty = true;
      if (timer !== null) clearTimeout(timer);
      timer = setTimeout(write, delay);
    },
    flush() {
      if (timer !== null) clearTimeout(timer);
      write();
    },
    cancel() {
      if (timer !== null) clearTimeout(timer);
      timer = null;
      dirty = false;
    },
  };
}
