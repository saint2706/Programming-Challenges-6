import { deserialize, reconcile, serialize } from './order.js';

export const STORAGE_KEY = 'photo-album:order';

/**
 * localStorage can be missing (privacy modes, sandboxed iframes), full, or
 * throw on access, so every call is guarded and failure degrades to "no
 * persistence" instead of breaking the album.
 */
export function loadOrder(storage, canonical) {
  let raw = null;
  try {
    raw = storage?.getItem(STORAGE_KEY) ?? null;
  } catch {
    raw = null;
  }
  return reconcile(deserialize(raw), canonical);
}

export function saveOrder(storage, order) {
  try {
    storage?.setItem(STORAGE_KEY, serialize(order));
    return true;
  } catch {
    return false;
  }
}

export function clearOrder(storage) {
  try {
    storage?.removeItem(STORAGE_KEY);
  } catch {
    // nothing to clean up if storage is unavailable
  }
}
