// Pure, DOM-free order logic. Every function returns a new array and never
// mutates its input, so callers can compare before/after cheaply.

export const STORAGE_VERSION = 1;

/** Moves the item at `from` so it ends up at index `to` (both clamped). */
export function moveItem(order, from, to) {
  if (order.length === 0 || from < 0 || from >= order.length) return [...order];
  const target = Math.max(0, Math.min(order.length - 1, to));
  const next = [...order];
  const [item] = next.splice(from, 1);
  next.splice(target, 0, item);
  return next;
}

/** Moves `id` to index `to`. Unknown ids leave the order unchanged. */
export function moveById(order, id, to) {
  return moveItem(order, order.indexOf(id), to);
}

/** Moves `id` one step (delta = -1 earlier, +1 later), stopping at the ends. */
export function nudge(order, id, delta) {
  const from = order.indexOf(id);
  return from === -1 ? [...order] : moveItem(order, from, from + delta);
}

/**
 * Drop semantics for drag-and-drop: place `dragId` immediately before or after
 * `targetId`. The insertion index is computed *after* removing the dragged
 * item, which is what makes "drop on the right edge of my left neighbour" a
 * no-op instead of an off-by-one shuffle.
 */
export function placeRelativeTo(order, dragId, targetId, edge) {
  if (dragId === targetId) return [...order];
  if (!order.includes(dragId) || !order.includes(targetId)) return [...order];
  const rest = order.filter((id) => id !== dragId);
  const at = rest.indexOf(targetId) + (edge === 'after' || edge === 'right' ? 1 : 0);
  rest.splice(at, 0, dragId);
  return rest;
}

/**
 * Reconciles a possibly stale saved order with the current canonical id list:
 * unknown and duplicate ids are dropped, photos added since the order was
 * saved are appended in canonical order.
 */
export function reconcile(saved, canonical) {
  const known = new Set(canonical);
  const seen = new Set();
  const kept = [];
  for (const id of Array.isArray(saved) ? saved : []) {
    if (typeof id === 'string' && known.has(id) && !seen.has(id)) {
      seen.add(id);
      kept.push(id);
    }
  }
  return [...kept, ...canonical.filter((id) => !seen.has(id))];
}

export function isSameOrder(a, b) {
  return a.length === b.length && a.every((id, i) => id === b[i]);
}

export function serialize(order) {
  return JSON.stringify({ v: STORAGE_VERSION, order });
}

/** Returns the stored id array, or null if the payload is missing or unusable. */
export function deserialize(text) {
  if (typeof text !== 'string') return null;
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    return null;
  }
  if (!data || typeof data !== 'object' || data.v !== STORAGE_VERSION) return null;
  return Array.isArray(data.order) ? data.order.filter((id) => typeof id === 'string') : null;
}
