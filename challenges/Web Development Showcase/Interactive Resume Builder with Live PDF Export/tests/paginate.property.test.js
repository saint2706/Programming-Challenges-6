import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { paginate } from '../src/lib/paginate.js';

const EPS = 0.01;

/** One random region whose blocks always fit a page, so no block should ever be oversize. */
function regionArb(contentHeight) {
  const small = Math.max(2, Math.floor(contentHeight / 8));
  const rowArb = fc.integer({ min: 1, max: small });
  const blockArb = fc.record({
    keepWithNext: fc.boolean(),
    atomHeight: fc.integer({ min: 1, max: small }),
    rows: fc.option(fc.array(rowArb, { minLength: 1, maxLength: 14 }), { nil: undefined }),
    chrome: fc.integer({ min: 0, max: Math.floor(small / 2) }),
  });
  return fc.array(blockArb, { maxLength: 40 }).map((specs) => {
    let chain = 0;
    return specs.map((spec, i) => {
      // At most two consecutive keepWithNext headings, so a chain always fits a page. A flag on a
      // list block is passed through on purpose: the paginator must ignore it.
      const atomic = !spec.rows;
      const keep = atomic ? spec.keepWithNext && chain < 2 : spec.keepWithNext;
      chain = atomic && keep ? chain + 1 : 0;
      const block = { id: `b${i}`, keepWithNext: keep };
      if (spec.rows) {
        block.rows = spec.rows.map((height) => ({ height }));
        block.chrome = spec.chrome;
      } else block.height = spec.atomHeight;
      return block;
    });
  });
}

const scenarioArb = fc
  .record({
    contentHeight: fc.integer({ min: 120, max: 900 }),
    orphans: fc.integer({ min: 1, max: 3 }),
    widows: fc.integer({ min: 1, max: 3 }),
  })
  .chain((options) =>
    fc.record({
      options: fc.constant(options),
      main: regionArb(options.contentHeight),
      side: regionArb(options.contentHeight),
    }),
  );

const rowHeights = (block) => (block.rows ? block.rows.map((r) => r.height) : [block.height]);

function fragmentsOf(result, regionId) {
  return result.pages.flatMap((page, pageIndex) =>
    (page.regions[regionId] ?? []).map((f) => ({ ...f, pageIndex })),
  );
}

describe('paginate (property tests)', () => {
  it('places every block and row exactly once, in order', () => {
    fc.assert(
      fc.property(scenarioArb, ({ options, main, side }) => {
        const regions = [
          { id: 'main', blocks: main },
          { id: 'side', blocks: side },
        ];
        const result = paginate(regions, options);
        for (const region of regions) {
          const frags = fragmentsOf(result, region.id);
          const order = [];
          for (const f of frags) if (order.at(-1) !== f.blockId) order.push(f.blockId);
          expect(order).toEqual(region.blocks.map((b) => b.id));
          for (const block of region.blocks) {
            const mine = frags.filter((f) => f.blockId === block.id);
            if (!block.rows) {
              expect(mine).toHaveLength(1);
              continue;
            }
            expect(mine[0].rowStart).toBe(0);
            expect(mine.at(-1).rowEnd).toBe(block.rows.length);
            for (let i = 1; i < mine.length; i++) expect(mine[i].rowStart).toBe(mine[i - 1].rowEnd);
          }
        }
      }),
      { numRuns: 300 },
    );
  });

  it('never lets a page exceed the content height', () => {
    fc.assert(
      fc.property(scenarioArb, ({ options, main }) => {
        const blocks = new Map(main.map((b) => [b.id, b]));
        const result = paginate([{ id: 'main', blocks: main }], options);
        expect(result.oversize).toEqual([]);
        for (const page of result.pages) {
          let used = 0;
          for (const f of page.regions.main) {
            const block = blocks.get(f.blockId);
            const heights = rowHeights(block);
            const slice = f.rowStart === null ? heights : heights.slice(f.rowStart, f.rowEnd);
            used += slice.reduce((a, b) => a + b, 0) + (block.rows ? block.chrome : 0);
          }
          expect(used).toBeLessThanOrEqual(options.contentHeight + EPS);
        }
      }),
      { numRuns: 300 },
    );
  });

  it('never ends a page on a keepWithNext block that has something after it', () => {
    fc.assert(
      fc.property(scenarioArb, ({ options, main }) => {
        const blocks = new Map(main.map((b) => [b.id, b]));
        const last = main.at(-1);
        const result = paginate([{ id: 'main', blocks: main }], options);
        for (const page of result.pages) {
          const tail = page.regions.main.at(-1);
          if (!tail) continue;
          const block = blocks.get(tail.blockId);
          if (block.keepWithNext && !block.rows && block !== last) {
            throw new Error(`page ends on keepWithNext block ${block.id}`);
          }
        }
      }),
      { numRuns: 300 },
    );
  });

  it('respects orphans and widows whenever a list is split', () => {
    fc.assert(
      fc.property(scenarioArb, ({ options, main }) => {
        const result = paginate([{ id: 'main', blocks: main }], options);
        const frags = fragmentsOf(result, 'main');
        for (const block of main.filter((b) => b.rows)) {
          const mine = frags.filter((f) => f.blockId === block.id);
          mine.forEach((f, i) => {
            const count = f.rowEnd - f.rowStart;
            if (i < mine.length - 1) expect(count).toBeGreaterThanOrEqual(options.orphans);
            if (i > 0) expect(count).toBeGreaterThanOrEqual(options.widows);
          });
        }
      }),
      { numRuns: 300 },
    );
  });

  it('is deterministic and page count is the longest region', () => {
    fc.assert(
      fc.property(scenarioArb, ({ options, main, side }) => {
        const regions = [
          { id: 'main', blocks: main },
          { id: 'side', blocks: side },
        ];
        const a = paginate(regions, options);
        expect(paginate(regions, options)).toEqual(a);
        const alone = (region) => paginate([region], options).pageCount;
        expect(a.pageCount).toBe(Math.max(alone(regions[0]), alone(regions[1])));
      }),
      { numRuns: 200 },
    );
  });
});
