// The grid uses tiny implicit rows (ROW_UNIT px) and each tile spans however
// many it needs. Because every photo's aspect ratio is known up front, the
// span is computed from the column width alone -- no waiting on image loads,
// so the layout never shifts.

export const ROW_UNIT = 4;

/** Rows a tile must span: its rendered height plus the trailing gap. */
export function rowSpan({ width, height, columnWidth, gap, rowUnit = ROW_UNIT }) {
  if (!(columnWidth > 0) || !(width > 0) || !(height > 0)) return 1;
  return Math.max(1, Math.ceil((columnWidth * (height / width) + gap) / rowUnit));
}

/** Applies spans to every tile in `grid`. `photoById` maps id -> {width, height}. */
export function layoutMasonry(grid, photoById) {
  const first = grid.firstElementChild;
  if (!first) return;
  const columnWidth = first.getBoundingClientRect().width;
  const gap = parseFloat(getComputedStyle(grid).columnGap) || 16;
  for (const tile of grid.children) {
    const photo = photoById.get(tile.dataset.id);
    if (!photo) continue;
    tile.style.gridRowEnd = `span ${rowSpan({ ...photo, columnWidth, gap })}`;
  }
}
