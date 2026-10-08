import { describe, expect, it } from 'vitest';
import { paginate } from '../src/lib/paginate.js';

const atom = (id, height, extra = {}) => ({ id, height, ...extra });
const list = (id, rowHeights, extra = {}) => ({
  id,
  rows: rowHeights.map((height) => ({ height })),
  ...extra,
});
const main = (...blocks) => [{ id: 'main', blocks }];
const opts = (contentHeight = 100) => ({ contentHeight });
const ids = (page, region = 'main') => page.regions[region].map((f) => f.blockId);
const rows = (page, blockId, region = 'main') => {
  const f = page.regions[region].find((x) => x.blockId === blockId);
  return f && [f.rowStart, f.rowEnd];
};

describe('atomic blocks', () => {
  it('keeps everything on one page when it fits', () => {
    const r = paginate(main(atom('a', 30), atom('b', 30), atom('c', 30)), opts());
    expect(r.pageCount).toBe(1);
    expect(ids(r.pages[0])).toEqual(['a', 'b', 'c']);
    expect(r.oversize).toEqual([]);
  });

  it('fills a page exactly without spilling', () => {
    const r = paginate(main(atom('a', 50), atom('b', 50)), opts());
    expect(r.pageCount).toBe(1);
  });

  it('starts a new page when the next block does not fit', () => {
    const r = paginate(main(atom('a', 60), atom('b', 60)), opts());
    expect(r.pageCount).toBe(2);
    expect(ids(r.pages[0])).toEqual(['a']);
    expect(ids(r.pages[1])).toEqual(['b']);
  });

  it('marks fragments of atomic blocks with null row bounds', () => {
    const r = paginate(main(atom('a', 10)), opts());
    expect(r.pages[0].regions.main[0]).toEqual({ blockId: 'a', rowStart: null, rowEnd: null });
  });

  it('places an oversize block alone on its page and reports it', () => {
    const r = paginate(main(atom('a', 20), atom('big', 150), atom('b', 20)), opts());
    expect(r.oversize).toEqual(['big']);
    expect(ids(r.pages[0])).toEqual(['a']);
    expect(ids(r.pages[1])).toEqual(['big']);
    expect(ids(r.pages[2])).toEqual(['b']);
  });
});

describe('keepWithNext', () => {
  it('moves a heading to the next page together with the block that follows it', () => {
    const r = paginate(main(atom('a', 70), atom('h', 15, { keepWithNext: true }), atom('p', 30)), opts());
    expect(ids(r.pages[0])).toEqual(['a']);
    expect(ids(r.pages[1])).toEqual(['h', 'p']);
  });

  it('keeps a whole chain of headings with the following block', () => {
    const r = paginate(
      main(
        atom('a', 60),
        atom('h1', 10, { keepWithNext: true }),
        atom('h2', 10, { keepWithNext: true }),
        atom('p', 25),
      ),
      opts(),
    );
    expect(ids(r.pages[0])).toEqual(['a']);
    expect(ids(r.pages[1])).toEqual(['h1', 'h2', 'p']);
  });

  it('keeps a heading with the first two rows of a list', () => {
    const fits = paginate(
      main(atom('a', 50), atom('h', 10, { keepWithNext: true }), list('l', [20, 20, 20, 20, 20])),
      opts(),
    );
    expect(ids(fits.pages[0])).toEqual(['a', 'h', 'l']);
    expect(rows(fits.pages[0], 'l')).toEqual([0, 2]);
    expect(rows(fits.pages[1], 'l')).toEqual([2, 5]);

    const tooLow = paginate(
      main(atom('a', 60), atom('h', 10, { keepWithNext: true }), list('l', [20, 20, 20, 20, 20])),
      opts(),
    );
    expect(ids(tooLow.pages[0])).toEqual(['a']);
    expect(ids(tooLow.pages[1])).toEqual(['h', 'l']);
  });

  it('ignores keepWithNext on a list block, which splits normally', () => {
    const r = paginate(
      main(atom('a', 60), list('l', [20, 20, 20, 20], { keepWithNext: true }), atom('b', 10)),
      opts(),
    );
    expect(ids(r.pages[0])).toEqual(['a', 'l']);
    expect(rows(r.pages[0], 'l')).toEqual([0, 2]);
    expect(rows(r.pages[1], 'l')).toEqual([2, 4]);
  });

  it('places a trailing keepWithNext block that has nothing after it', () => {
    const r = paginate(main(atom('a', 30), atom('h', 10, { keepWithNext: true })), opts());
    expect(ids(r.pages[0])).toEqual(['a', 'h']);
  });
});

describe('splitting lists', () => {
  it('splits between rows when a list runs past the page', () => {
    const r = paginate(main(list('l', [30, 30, 30, 30, 30])), opts());
    expect(r.pageCount).toBe(2);
    expect(rows(r.pages[0], 'l')).toEqual([0, 3]);
    expect(rows(r.pages[1], 'l')).toEqual([3, 5]);
  });

  it('leaves at least two rows (widows) for the next page', () => {
    const r = paginate(main(list('l', [30, 30, 30, 30])), opts());
    expect(rows(r.pages[0], 'l')).toEqual([0, 2]);
    expect(rows(r.pages[1], 'l')).toEqual([2, 4]);
  });

  it('moves the whole list when fewer than two rows (orphans) would fit', () => {
    const r = paginate(main(atom('a', 70), list('l', [20, 20, 20, 20])), opts());
    expect(ids(r.pages[0])).toEqual(['a']);
    expect(rows(r.pages[1], 'l')).toEqual([0, 4]);
  });

  it('never splits a list shorter than orphans + widows', () => {
    const r = paginate(main(atom('a', 40), list('l', [30, 30, 30])), opts());
    expect(ids(r.pages[0])).toEqual(['a']);
    expect(rows(r.pages[1], 'l')).toEqual([0, 3]);
  });

  it('splits a list taller than a page at row boundaries without losing rows', () => {
    const r = paginate(main(list('l', [40, 40, 40, 40, 40])), opts());
    const covered = r.pages.flatMap((p) => p.regions.main).map((f) => [f.rowStart, f.rowEnd]);
    expect(covered[0][0]).toBe(0);
    expect(covered.at(-1)[1]).toBe(5);
    for (let i = 1; i < covered.length; i++) expect(covered[i][0]).toBe(covered[i - 1][1]);
  });

  it('honours custom orphans and widows', () => {
    const r = paginate(main(list('l', [20, 20, 20, 20, 20, 20, 20, 20])), {
      contentHeight: 100,
      orphans: 3,
      widows: 3,
    });
    expect(rows(r.pages[0], 'l')).toEqual([0, 5]);
    expect(rows(r.pages[1], 'l')).toEqual([5, 8]);
  });
});

describe('list chrome (the space a list container takes beyond its rows, charged once per fragment)', () => {
  it('counts the chrome of a list that fits', () => {
    // 3 rows of 20 + 30 chrome = 90
    const r = paginate(main(atom('a', 20), list('l', [20, 20, 20], { chrome: 30 })), opts(100));
    expect(r.pageCount).toBe(2);
    expect(ids(r.pages[0])).toEqual(['a']);
    expect(rows(r.pages[1], 'l')).toEqual([0, 3]);
  });

  it('charges the chrome again for every fragment of a split list', () => {
    // each fragment costs 10 + its rows. Page 1 fits 4 rows of 20 (10 + 80 = 90), page 2 the rest.
    const r = paginate(main(list('l', Array(8).fill(20), { chrome: 10 })), opts(100));
    expect(rows(r.pages[0], 'l')).toEqual([0, 4]);
    expect(rows(r.pages[1], 'l')).toEqual([4, 8]);
    // without chrome the same list splits 5 + 3
    const plain = paginate(main(list('l', Array(8).fill(20))), opts(100));
    expect(rows(plain.pages[0], 'l')).toEqual([0, 5]);
  });

  it('includes the chrome when keeping a heading with the first rows', () => {
    // heading 10 + chrome 30 + two rows of 20 = 80 > the 70 left, so heading and list move together
    const r = paginate(
      main(atom('a', 30), atom('h', 10, { keepWithNext: true }), list('l', [20, 20, 20, 20], { chrome: 30 })),
      opts(100),
    );
    expect(ids(r.pages[0])).toEqual(['a']);
    expect(ids(r.pages[1])).toEqual(['h', 'l']);
  });

  it('treats a missing chrome as zero', () => {
    expect(paginate(main(list('l', [50, 50])), opts(100)).pageCount).toBe(1);
  });

  it('rejects a bad chrome', () => {
    expect(() => paginate(main(list('l', [10, 10], { chrome: -1 })), opts())).toThrow(RangeError);
    expect(() => paginate(main(list('l', [10, 10], { chrome: Number.NaN })), opts())).toThrow(RangeError);
  });
});

describe('regions', () => {
  it('paginates regions independently and counts the longest', () => {
    const r = paginate(
      [
        { id: 'main', blocks: [atom('m1', 60), atom('m2', 60), atom('m3', 60)] },
        { id: 'side', blocks: [atom('s1', 40)] },
      ],
      opts(),
    );
    expect(r.pageCount).toBe(3);
    expect(ids(r.pages[0], 'side')).toEqual(['s1']);
    expect(r.pages[1].regions.side).toEqual([]);
    expect(r.pages[2].regions.side).toEqual([]);
  });

  it('gives an empty region one empty page', () => {
    const r = paginate(main(), opts());
    expect(r.pageCount).toBe(1);
    expect(r.pages[0].regions.main).toEqual([]);
  });

  it('handles no regions at all', () => {
    const r = paginate([], opts());
    expect(r.pageCount).toBe(1);
    expect(r.pages[0].regions).toEqual({});
  });
});

describe('contract', () => {
  it('is deterministic and does not mutate its input', () => {
    const regions = main(atom('a', 60), atom('h', 10, { keepWithNext: true }), list('l', [20, 20, 20, 20]));
    const snapshot = JSON.stringify(regions);
    expect(paginate(regions, opts())).toEqual(paginate(regions, opts()));
    expect(JSON.stringify(regions)).toBe(snapshot);
  });

  it('rejects unusable options', () => {
    expect(() => paginate(main(), { contentHeight: 0 })).toThrow(RangeError);
    expect(() => paginate(main(), { contentHeight: Number.NaN })).toThrow(RangeError);
    expect(() => paginate(main(), { contentHeight: 100, orphans: 0 })).toThrow(RangeError);
    expect(() => paginate(main(), { contentHeight: 100, widows: 0 })).toThrow(RangeError);
  });

  it('rejects blocks with unusable heights', () => {
    expect(() => paginate(main(atom('a', -1)), opts())).toThrow(RangeError);
    expect(() => paginate(main(atom('a', Number.NaN)), opts())).toThrow(RangeError);
    expect(() => paginate(main(list('l', [10, Number.NaN])), opts())).toThrow(RangeError);
  });
});
