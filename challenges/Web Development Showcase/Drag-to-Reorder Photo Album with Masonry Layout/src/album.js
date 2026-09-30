import { isSameOrder, moveById, nudge, placeRelativeTo } from './order.js';
import { layoutMasonry } from './masonry.js';
import { clearOrder, loadOrder, saveOrder } from './storage.js';

const FLIP_MS = 220;

/**
 * Owns the album DOM, keyboard reordering, persistence and layout. Pointer
 * drag-and-drop is wired separately (dnd.js) and only calls `commitDrop`.
 * All text from photo data goes through textContent / setAttribute, never
 * innerHTML.
 */
export function createAlbum({
  grid,
  photos,
  storage = null,
  announce = () => {},
  prefersReducedMotion = () => globalThis.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false,
}) {
  const canonical = photos.map((p) => p.id);
  const photoById = new Map(photos.map((p) => [p.id, p]));
  const tiles = new Map();
  let order = loadOrder(storage, canonical);
  let grabbed = null; // { id, origin } while a keyboard reorder is in progress
  let moving = false; // true while we re-insert nodes (which blurs the focused tile)

  const title = (id) => photoById.get(id).title;
  const position = (id) => `Position ${order.indexOf(id) + 1} of ${order.length}.`;

  for (const photo of photos) tiles.set(photo.id, buildTile(photo));

  function buildTile(photo) {
    const li = document.createElement('li');
    li.className = 'tile';
    li.dataset.id = photo.id;
    li.tabIndex = 0;
    li.setAttribute('aria-roledescription', 'sortable photo');
    li.setAttribute('aria-describedby', 'album-help');

    const figure = document.createElement('figure');
    const img = document.createElement('img');
    img.src = photo.src;
    img.alt = photo.alt;
    img.width = photo.width;
    img.height = photo.height;
    img.draggable = false; // the tile is the drag source, not the image
    img.decoding = 'async';
    const caption = document.createElement('figcaption');
    const name = document.createElement('span');
    name.className = 'tile-title';
    name.textContent = photo.title;
    const credit = document.createElement('span');
    credit.className = 'tile-credit';
    credit.textContent = `by ${photo.photographer}`;
    caption.append(name, credit);
    figure.append(img, caption);
    li.append(figure);

    li.addEventListener('keydown', (event) => onKeyDown(event, photo.id));
    li.addEventListener('focusout', () => {
      if (grabbed?.id === photo.id && !moving) drop();
    });
    return li;
  }

  function relayout() {
    layoutMasonry(grid, photoById);
  }

  /** Re-inserts tiles in `order`, relayouts, and animates displaced tiles (FLIP). */
  function paint(focusId) {
    const animate = !prefersReducedMotion();
    const before = animate ? new Map([...tiles].map(([id, el]) => [id, el.getBoundingClientRect()])) : null;

    moving = true;
    for (const id of order) grid.append(tiles.get(id));
    order.forEach((id, i) => {
      tiles.get(id).setAttribute('aria-label', `${title(id)}, by ${photoById.get(id).photographer}. ${i + 1} of ${order.length}.`);
    });
    relayout();
    if (focusId) tiles.get(focusId).focus({ preventScroll: false });
    moving = false;

    if (before) {
      for (const [id, el] of tiles) {
        if (typeof el.animate !== 'function') return;
        const from = before.get(id);
        const to = el.getBoundingClientRect();
        const dx = from.left - to.left;
        const dy = from.top - to.top;
        if (dx || dy) {
          el.animate([{ transform: `translate(${dx}px, ${dy}px)` }, { transform: 'none' }], {
            duration: FLIP_MS,
            easing: 'cubic-bezier(0.2, 0, 0, 1)',
          });
        }
      }
    }
  }

  function setOrder(next, { persist = true, focusId = null } = {}) {
    const changed = !isSameOrder(order, next);
    order = next;
    if (changed || !grid.childElementCount) paint(focusId);
    else if (focusId) tiles.get(focusId).focus();
    if (persist) saveOrder(storage, order);
    return changed;
  }

  // ---- keyboard reordering -------------------------------------------------

  function grab(id) {
    if (grabbed) return;
    grabbed = { id, origin: [...order] };
    tiles.get(id).dataset.grabbed = 'true';
    announce(`Grabbed ${title(id)}. ${position(id)} Arrow keys move it, Space or Enter drops it, Escape cancels.`);
  }

  function moveGrabbed(step) {
    if (!grabbed) return;
    const { id } = grabbed;
    const next =
      step === 'start' ? moveById(order, id, 0) : step === 'end' ? moveById(order, id, order.length - 1) : nudge(order, id, step);
    if (setOrder(next, { persist: false, focusId: id })) announce(`Moved ${title(id)}. ${position(id)}`);
    else announce(`${title(id)} is already at the ${order.indexOf(id) === 0 ? 'start' : 'end'}. ${position(id)}`);
  }

  function drop() {
    if (!grabbed) return;
    const { id, origin } = grabbed;
    delete tiles.get(id).dataset.grabbed;
    grabbed = null;
    saveOrder(storage, order);
    announce(isSameOrder(origin, order) ? `Dropped ${title(id)} in place. ${position(id)}` : `Dropped ${title(id)}. ${position(id)}`);
  }

  function cancel() {
    if (!grabbed) return;
    const { id, origin } = grabbed;
    delete tiles.get(id).dataset.grabbed;
    grabbed = null;
    setOrder(origin, { persist: false, focusId: id });
    announce(`Reorder cancelled. ${title(id)} is back at ${position(id).toLowerCase()}`);
  }

  function onKeyDown(event, id) {
    if (event.target !== tiles.get(id) || event.altKey || event.ctrlKey || event.metaKey) return;
    const { key } = event;
    if (key === ' ' || key === 'Enter') {
      event.preventDefault();
      if (grabbed?.id === id) drop();
      else grab(id);
      return;
    }
    if (grabbed?.id !== id) return;
    const steps = { ArrowLeft: -1, ArrowUp: -1, ArrowRight: 1, ArrowDown: 1, Home: 'start', End: 'end' };
    if (key in steps) {
      event.preventDefault();
      moveGrabbed(steps[key]);
    } else if (key === 'Escape') {
      event.preventDefault();
      cancel();
    }
  }

  // ---- pointer drop (called by dnd.js) ------------------------------------

  function commitDrop(dragId, targetId, edge) {
    const next = placeRelativeTo(order, dragId, targetId, edge);
    if (setOrder(next, { persist: true })) {
      announce(`Moved ${title(dragId)}. ${position(dragId)}`);
    }
  }

  function reset() {
    if (grabbed) cancel();
    clearOrder(storage);
    setOrder([...canonical], { persist: false });
    announce('Album order reset to the default.');
  }

  paint(null);
  const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(relayout) : null;
  observer?.observe(grid);

  return {
    tiles,
    getOrder: () => [...order],
    isGrabbed: () => grabbed !== null,
    isDefaultOrder: () => isSameOrder(order, canonical),
    commitDrop,
    reset,
    relayout,
    destroy: () => observer?.disconnect(),
  };
}
