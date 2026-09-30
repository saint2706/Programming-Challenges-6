import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { draggable, dropTargetForElements, monitorForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { attachClosestEdge, extractClosestEdge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/closest-edge';

/**
 * Wires Pragmatic drag and drop onto every tile. The library is a thin layer
 * over native HTML5 drag events; the closest-edge hitbox tells us whether the
 * pointer is on the left or right half of a tile, which becomes "insert
 * before" / "insert after" and drives the drop indicator.
 */
export function enableDragAndDrop(album) {
  const cleanups = [];

  for (const [id, tile] of album.tiles) {
    cleanups.push(
      combine(
        draggable({
          element: tile,
          getInitialData: () => ({ type: 'photo', id }),
          onDragStart: () => {
            tile.dataset.dragging = 'true';
          },
          onDrop: () => {
            delete tile.dataset.dragging;
          },
        }),
        dropTargetForElements({
          element: tile,
          canDrop: ({ source }) => source.data.type === 'photo' && source.data.id !== id,
          getData: ({ input, element }) => attachClosestEdge({ id }, { input, element, allowedEdges: ['left', 'right'] }),
          onDrag: ({ self }) => {
            tile.dataset.dropEdge = extractClosestEdge(self.data) ?? '';
          },
          onDragLeave: () => {
            delete tile.dataset.dropEdge;
          },
          onDrop: () => {
            delete tile.dataset.dropEdge;
          },
        }),
      ),
    );
  }

  cleanups.push(
    monitorForElements({
      canMonitor: ({ source }) => source.data.type === 'photo',
      onDrop: ({ source, location }) => {
        const target = location.current.dropTargets[0];
        if (!target) return;
        album.commitDrop(source.data.id, target.data.id, extractClosestEdge(target.data));
      },
    }),
  );

  return () => cleanups.forEach((fn) => fn());
}
