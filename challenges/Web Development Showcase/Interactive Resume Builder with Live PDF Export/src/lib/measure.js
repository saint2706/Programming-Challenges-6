/**
 * Block heights for the paginator, with a cache so a keystroke re-measures only the block it
 * changed. This module is DOM-free: it is handed a `measure(blocks)` function (the Preview
 * component's real one renders the blocks into a hidden container and reads their heights; tests
 * pass a fake) and decides *what* needs measuring.
 *
 * `measure(blocks)` must return, synchronously and in order, `{ height }` for an atomic block and
 * `{ height, rows: number[] }` (one height per row) for a list block.
 */

/**
 * Thrown by a `measure` function when the tree it measures in is not laid out right now (for
 * example hidden by a stylesheet). Callers keep what they have instead of caching zero heights.
 */
export class MeasureUnavailable extends Error {
  constructor(message = 'the measuring tree is not laid out') {
    super(message);
    this.name = 'MeasureUnavailable';
  }
}

export function heightKey(layoutKey, block) {
  return `${layoutKey}|${block.id}|${block.hash}`;
}

function validate(block, m) {
  const bad = (h) => !Number.isFinite(h) || h < 0;
  if (!m || bad(m.height)) throw new RangeError(`bad height measured for block "${block.id}"`);
  if (block.rows) {
    if (!Array.isArray(m.rows) || m.rows.length !== block.rows.length) {
      throw new RangeError(`block "${block.id}" needs ${block.rows.length} rows measured`);
    }
    if (m.rows.some(bad)) throw new RangeError(`bad row height measured for block "${block.id}"`);
  }
}

/**
 * The block as the paginator wants it: id, keepWithNext and heights only. A list's `chrome` is the
 * space its container takes beyond its rows (the padding above the first row), measured as the
 * container's height minus the sum of its row heights.
 */
function sized(block, m) {
  if (block.rows) {
    const chrome = Math.max(0, m.height - m.rows.reduce((total, h) => total + h, 0));
    return {
      id: block.id,
      keepWithNext: block.keepWithNext,
      rows: block.rows.map((row, i) => ({ id: row.id, height: m.rows[i] })),
      chrome,
    };
  }
  return { id: block.id, keepWithNext: block.keepWithNext, height: m.height };
}

export function createMeasurer({ measure, maxIdle = 2000 }) {
  const cache = new Map();
  const measurer = {
    cache,
    stats: { measured: 0, reused: 0 },
    /**
     * Heights for every block in `regions`, measuring only what is not cached for `layoutKey`
     * (a string covering everything that changes block sizes: template, page size, font, scale).
     * Returns regions of sized blocks, ready for `paginate`.
     */
    resolve(regions, layoutKey) {
      const blocks = regions.flatMap((r) => r.blocks);
      const missing = blocks.filter((b) => !cache.has(heightKey(layoutKey, b)));
      if (missing.length > 0) {
        const results = measure(missing);
        if (!Array.isArray(results) || results.length !== missing.length) {
          throw new RangeError('measure() must return one result per block');
        }
        missing.forEach((b, i) => {
          validate(b, results[i]);
          cache.set(heightKey(layoutKey, b), results[i]);
        });
      }
      measurer.stats = { measured: missing.length, reused: blocks.length - missing.length };

      if (cache.size - blocks.length > maxIdle) {
        const live = new Set(blocks.map((b) => heightKey(layoutKey, b)));
        for (const key of cache.keys()) if (!live.has(key)) cache.delete(key);
      }
      return regions.map((region) => ({
        id: region.id,
        blocks: region.blocks.map((b) => sized(b, cache.get(heightKey(layoutKey, b)))),
      }));
    },
  };
  return measurer;
}
