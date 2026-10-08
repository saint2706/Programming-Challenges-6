import { describe, expect, it } from 'vitest';
import { buildRegions } from '../src/lib/blocks.js';
import { MeasureUnavailable, createMeasurer, heightKey } from '../src/lib/measure.js';
import { setPath } from '../src/lib/model.js';
import { paginate } from '../src/lib/paginate.js';
import { longResume } from '../src/lib/sample-long.js';

/** A stand-in for the DOM: every block is 10px tall, every list row 5px. */
function fakeMeasure() {
  const calls = [];
  const measure = (blocks) => {
    calls.push(blocks.map((b) => b.id));
    return blocks.map((b) => (b.rows ? { height: b.rows.length * 5, rows: b.rows.map(() => 5) } : { height: 10 }));
  };
  return { measure, calls };
}

describe('createMeasurer', () => {
  it('measures every block once and serves repeats from the cache', () => {
    const { measure, calls } = fakeMeasure();
    const measurer = createMeasurer({ measure });
    const regions = buildRegions(longResume());
    const total = regions.flatMap((r) => r.blocks).length;

    measurer.resolve(regions, 'L1');
    expect(measurer.stats).toEqual({ measured: total, reused: 0 });
    measurer.resolve(regions, 'L1');
    expect(measurer.stats).toEqual({ measured: 0, reused: total });
    expect(calls).toHaveLength(1);
  });

  it('re-measures only the block whose content changed', () => {
    const { measure, calls } = fakeMeasure();
    const measurer = createMeasurer({ measure });
    const doc = longResume();
    measurer.resolve(buildRegions(doc), 'L1');
    measurer.resolve(buildRegions(setPath(doc, 'work.1.position', 'Principal Engineer')), 'L1');
    expect(calls.at(-1)).toEqual(['work.1.header']);
    expect(measurer.stats.measured).toBe(1);
  });

  it('re-measures everything when the layout key changes', () => {
    const { measure } = fakeMeasure();
    const measurer = createMeasurer({ measure });
    const regions = buildRegions(longResume());
    const total = regions.flatMap((r) => r.blocks).length;
    measurer.resolve(regions, 'a4|inter|1');
    measurer.resolve(regions, 'a4|inter|0.9');
    expect(measurer.stats.measured).toBe(total);
  });

  it('returns blocks carrying heights (and row heights) ready for the paginator', () => {
    const { measure } = fakeMeasure();
    const measurer = createMeasurer({ measure });
    const resolved = measurer.resolve(buildRegions(longResume()), 'L1');
    const list = resolved[0].blocks.find((b) => b.id === 'work.0.bullets');
    expect(list.rows).toHaveLength(5);
    expect(list.rows[0]).toEqual({ id: 'work.0.highlights.0', height: 5 });
    expect(resolved[0].blocks[0]).toEqual({ id: 'header', keepWithNext: false, height: 10 });

    const result = paginate(resolved, { contentHeight: 200 });
    expect(result.pageCount).toBeGreaterThan(1);
    expect(result.oversize).toEqual([]);
  });

  it('derives the chrome of a list: its height beyond the sum of its rows', () => {
    const measurer = createMeasurer({
      measure: (blocks) => blocks.map((b) => (b.rows ? { height: b.rows.length * 5 + 7, rows: b.rows.map(() => 5) } : { height: 10 })),
    });
    const resolved = measurer.resolve(buildRegions(longResume()), 'L1');
    const list = resolved[0].blocks.find((b) => b.id === 'work.0.bullets');
    expect(list.chrome).toBe(7);
    expect(resolved[0].blocks[0].chrome).toBeUndefined();
  });

  it('never reports a negative chrome', () => {
    const measurer = createMeasurer({
      measure: (blocks) => blocks.map((b) => (b.rows ? { height: 1, rows: b.rows.map(() => 5) } : { height: 10 })),
    });
    const resolved = measurer.resolve(buildRegions(longResume()), 'L1');
    expect(resolved[0].blocks.find((b) => b.id === 'work.0.bullets').chrome).toBe(0);
  });

  it('rejects a measurement that is not a finite height or has the wrong row count', () => {
    const regions = buildRegions(longResume());
    const nan = createMeasurer({ measure: (blocks) => blocks.map(() => ({ height: Number.NaN })) });
    expect(() => nan.resolve(regions, 'L')).toThrow(RangeError);

    const short = createMeasurer({
      measure: (blocks) => blocks.map((b) => (b.rows ? { height: 5, rows: [5] } : { height: 10 })),
    });
    expect(() => short.resolve(regions, 'L')).toThrow(/rows/);

    const fewer = createMeasurer({ measure: () => [] });
    expect(() => fewer.resolve(regions, 'L')).toThrow(RangeError);
  });

  it('forgets entries that are no longer in use once the cache grows large', () => {
    const { measure } = fakeMeasure();
    const measurer = createMeasurer({ measure, maxIdle: 20 });
    const doc = longResume();
    measurer.resolve(buildRegions(doc), 'L1');
    const live = measurer.cache.size;
    for (let i = 0; i < 40; i++) {
      measurer.resolve(buildRegions(setPath(doc, 'basics.name', `Name ${i}`)), 'L1');
    }
    expect(measurer.cache.size).toBeLessThan(live + 20 + 1);
  });

  it('lets MeasureUnavailable through without caching anything', () => {
    let available = false;
    const measurer = createMeasurer({
      measure: (blocks) => {
        if (!available) throw new MeasureUnavailable();
        return blocks.map((b) => (b.rows ? { height: 5, rows: b.rows.map(() => 5) } : { height: 10 }));
      },
    });
    const regions = buildRegions(longResume());
    expect(() => measurer.resolve(regions, 'L')).toThrow(MeasureUnavailable);
    expect(measurer.cache.size).toBe(0);
    available = true;
    expect(() => measurer.resolve(regions, 'L')).not.toThrow();
    expect(new MeasureUnavailable()).toBeInstanceOf(Error);
    expect(new MeasureUnavailable().name).toBe('MeasureUnavailable');
  });

  it('builds cache keys from layout, id and hash', () => {
    expect(heightKey('L', { id: 'x', hash: 'abc' })).toBe('L|x|abc');
  });
});
