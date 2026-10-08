/**
 * The deterministic paginator: block heights in, pages out. No DOM, no randomness, no clock.
 *
 * Input: `regions` (e.g. `main` and `sidebar`), each an ordered list of blocks:
 *   - atomic block:  `{ id, height, keepWithNext? }`
 *   - list block:    `{ id, rows: [{ height }], chrome? }`; it may split between rows. Its height
 *     is `chrome` (the container's own space: padding above the first row) plus its rows, and every
 *     fragment of a split list is a separate container, so every fragment pays `chrome` again.
 * A `keepWithNext` atomic block (a heading, an entry header) never ends a page: it travels with
 * the block after it and, when that block is a splittable list, with its first `orphans` rows.
 * `keepWithNext` is for small headings only; on a list block it is ignored.
 *
 * Output: `{ pages, pageCount, oversize }`. Each page is `{ regions: { [regionId]: fragments } }`
 * where a fragment is `{ blockId, rowStart, rowEnd }` (null bounds for atomic blocks, a half-open
 * row range for list blocks). `oversize` lists blocks that cannot fit even an empty page; they are
 * placed anyway (and clipped by the page) so the UI can warn about them.
 *
 * Regions are independent; the page count is the longest region's. Lists shorter than
 * `orphans + widows` rows are never split.
 */

const EPS = 0.01;

function check(regions, contentHeight, orphans, widows) {
  if (!Number.isFinite(contentHeight) || contentHeight <= 0) {
    throw new RangeError('contentHeight must be a positive number');
  }
  if (!Number.isInteger(orphans) || orphans < 1) throw new RangeError('orphans must be >= 1');
  if (!Number.isInteger(widows) || widows < 1) throw new RangeError('widows must be >= 1');
  const bad = (h) => !Number.isFinite(h) || h < 0;
  for (const region of regions) {
    for (const block of region.blocks) {
      const heights = block.rows ? [...block.rows.map((r) => r.height), block.chrome ?? 0] : [block.height];
      if (heights.some(bad)) throw new RangeError(`block "${block.id}" has an invalid height`);
    }
  }
}

const sum = (rows, from = 0, to = rows.length) => {
  let total = 0;
  for (let i = from; i < to; i++) total += rows[i].height;
  return total;
};

function paginateRegion(blocks, contentHeight, orphans, widows, oversize) {
  const pages = [[]];
  let y = 0;

  const newPage = () => {
    pages.push([]);
    y = 0;
  };
  const chromeOf = (b) => b.chrome ?? 0;
  const heightOf = (b) => (b.rows ? chromeOf(b) + sum(b.rows) : b.height);
  const splittable = (b) => Boolean(b.rows) && b.rows.length >= orphans + widows;
  const headHeight = (b) => (splittable(b) ? chromeOf(b) + sum(b.rows, 0, orphans) : heightOf(b));

  function placeFragment(block, start, end) {
    if (block.rows) {
      pages.at(-1).push({ blockId: block.id, rowStart: start, rowEnd: end });
      y += chromeOf(block) + sum(block.rows, start, end);
    } else {
      pages.at(-1).push({ blockId: block.id, rowStart: null, rowEnd: null });
      y += block.height;
    }
  }

  /** Place `block`, splitting a list between rows where the orphan/widow rules allow. */
  function place(block) {
    if (!splittable(block)) {
      const h = heightOf(block);
      if (y > 0 && y + h > contentHeight + EPS) newPage();
      if (h > contentHeight + EPS) oversize.add(block.id);
      placeFragment(block, 0, block.rows?.length);
      return;
    }
    const n = block.rows.length;
    let start = 0;
    while (start < n) {
      const remaining = n - start;
      let fit = 0;
      let used = chromeOf(block);
      while (fit < remaining && used + block.rows[start + fit].height <= contentHeight - y + EPS) {
        used += block.rows[start + fit].height;
        fit++;
      }
      if (fit === remaining) {
        placeFragment(block, start, n);
        return;
      }
      const take = Math.min(fit, remaining - widows);
      if (take >= orphans) {
        placeFragment(block, start, start + take);
        newPage();
        start += take;
      } else if (y > 0) {
        newPage();
      } else {
        // An empty page and still no valid split: rows are too tall. Take what fits (at least
        // one row) rather than loop forever, and flag a row that is taller than a page.
        const force = Math.max(1, Math.min(fit, remaining - 1));
        for (let i = start; i < start + force; i++) {
          if (chromeOf(block) + block.rows[i].height > contentHeight + EPS) oversize.add(block.id);
        }
        placeFragment(block, start, start + force);
        if (start + force < n) newPage();
        start += force;
      }
    }
  }

  let i = 0;
  while (i < blocks.length) {
    const chain = [];
    while (i < blocks.length && blocks[i].keepWithNext && !blocks[i].rows) chain.push(blocks[i++]);
    const anchor = i < blocks.length ? blocks[i++] : null;
    if (chain.length > 0) {
      const unit = chain.reduce((total, b) => total + heightOf(b), 0) + (anchor ? headHeight(anchor) : 0);
      if (y > 0 && y + unit > contentHeight + EPS) newPage();
      if (unit > contentHeight + EPS) for (const b of chain) oversize.add(b.id);
      for (const b of chain) placeFragment(b, 0, b.rows?.length);
    }
    if (anchor) place(anchor);
  }
  return pages;
}

/**
 * @param {{id: string, blocks: object[]}[]} regions
 * @param {{contentHeight: number, orphans?: number, widows?: number}} options
 */
export function paginate(regions, { contentHeight, orphans = 2, widows = 2 }) {
  check(regions, contentHeight, orphans, widows);
  const oversize = new Set();
  const perRegion = regions.map((region) => ({
    id: region.id,
    pages: paginateRegion(region.blocks, contentHeight, orphans, widows, oversize),
  }));
  const pageCount = Math.max(1, ...perRegion.map((r) => r.pages.length));
  const pages = Array.from({ length: pageCount }, (_, p) => ({
    regions: Object.fromEntries(perRegion.map((r) => [r.id, r.pages[p] ?? []])),
  }));
  return { pages, pageCount, oversize: [...oversize] };
}
