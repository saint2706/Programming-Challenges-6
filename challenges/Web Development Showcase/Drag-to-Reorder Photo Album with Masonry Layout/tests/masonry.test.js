import { describe, expect, it } from 'vitest';
import { ROW_UNIT, rowSpan } from '../src/masonry.js';

describe('rowSpan', () => {
  it('covers rendered height plus the gap', () => {
    // 240px wide, 3:2 landscape => 160px tall; (160 + 16) / 4 = 44 rows
    expect(rowSpan({ width: 300, height: 200, columnWidth: 240, gap: 16 })).toBe(44);
  });
  it('gives portrait photos more rows than landscape ones', () => {
    const portrait = rowSpan({ width: 533, height: 800, columnWidth: 240, gap: 16 });
    const landscape = rowSpan({ width: 800, height: 450, columnWidth: 240, gap: 16 });
    expect(portrait).toBeGreaterThan(landscape);
  });
  it('rounds up so tiles never overlap', () => {
    const span = rowSpan({ width: 800, height: 534, columnWidth: 241, gap: 16 });
    expect(span * ROW_UNIT).toBeGreaterThanOrEqual(241 * (534 / 800) + 16);
  });
  it('scales with the column width', () => {
    // 160px wide => 120px tall => 30 rows; doubling the column doubles the height
    const narrow = rowSpan({ width: 4, height: 3, columnWidth: 160, gap: 0 });
    const wide = rowSpan({ width: 4, height: 3, columnWidth: 320, gap: 0 });
    expect(narrow).toBe(30);
    expect(wide).toBe(60);
  });
  it('degrades to a single row for unmeasured or invalid input', () => {
    expect(rowSpan({ width: 800, height: 600, columnWidth: 0, gap: 16 })).toBe(1);
    expect(rowSpan({ width: 0, height: 600, columnWidth: 240, gap: 16 })).toBe(1);
    expect(rowSpan({ width: 800, height: NaN, columnWidth: 240, gap: 16 })).toBe(1);
  });
});
