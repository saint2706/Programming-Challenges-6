import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import {
  draggable,
  dropTargetForElements,
  monitorForElements,
} from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import {
  attachClosestEdge,
  extractClosestEdge,
} from '@atlaskit/pragmatic-drag-and-drop-hitbox/closest-edge';

/**
 * Pointer drag-and-drop for the section list, using Atlassian's pragmatic-drag-and-drop (a thin
 * layer over native HTML5 drag events, the same library as the photo album challenge). The
 * keyboard path (Alt+Up/Down and the move buttons) does not depend on any of this.
 */

/**
 * The new order after dropping `id` above (`'top'`) or below (`'bottom'`) `targetId`. Returns the
 * same array when nothing would change, so callers can skip a no-op update.
 */
export function moveRelative(order, id, targetId, edge) {
  const from = order.indexOf(id);
  if (from === -1 || id === targetId || !order.includes(targetId)) return order;
  const without = order.filter((item) => item !== id);
  const at = without.indexOf(targetId) + (edge === 'bottom' ? 1 : 0);
  const next = [...without.slice(0, at), id, ...without.slice(at)];
  return next.every((item, i) => item === order[i]) ? order : next;
}

/**
 * Make each `<li data-section>` in `list` draggable by its `.handle` and a drop target, and call
 * `onmove(id, targetId, edge)` on drop. Returns a cleanup function.
 */
export function enableSectionDnd(list, onmove) {
  const cleanups = [];
  for (const item of list.querySelectorAll('li[data-section]')) {
    const id = item.dataset.section;
    const handle = item.querySelector('.handle');
    if (!handle) continue;
    cleanups.push(
      combine(
        draggable({
          element: item,
          dragHandle: handle,
          getInitialData: () => ({ type: 'section', id }),
          onDragStart: () => {
            item.dataset.dragging = 'true';
          },
          onDrop: () => {
            delete item.dataset.dragging;
          },
        }),
        dropTargetForElements({
          element: item,
          canDrop: ({ source }) => source.data.type === 'section' && source.data.id !== id,
          getData: ({ input, element }) =>
            attachClosestEdge({ id }, { input, element, allowedEdges: ['top', 'bottom'] }),
          onDrag: ({ self }) => {
            item.dataset.dropEdge = extractClosestEdge(self.data) ?? '';
          },
          onDragLeave: () => {
            delete item.dataset.dropEdge;
          },
          onDrop: () => {
            delete item.dataset.dropEdge;
          },
        }),
      ),
    );
  }
  cleanups.push(
    monitorForElements({
      canMonitor: ({ source }) => source.data.type === 'section',
      onDrop: ({ source, location }) => {
        const target = location.current.dropTargets[0];
        if (!target) return;
        onmove(source.data.id, target.data.id, extractClosestEdge(target.data));
      },
    }),
  );
  return () => cleanups.forEach((cleanup) => cleanup());
}
