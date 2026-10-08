import { describe, expect, it } from 'vitest';
import { MARGIN_MM, PAGE_SAFETY_PX, pageCss, pageMetrics } from '../src/lib/page.js';

describe('pageMetrics', () => {
  it('computes A4 in CSS pixels at 96 dpi', () => {
    const m = pageMetrics('a4');
    expect(m.widthPx).toBeCloseTo(793.7, 1);
    expect(m.heightPx).toBeCloseTo(1122.52, 1);
    const margins = ((MARGIN_MM.top + MARGIN_MM.bottom) * 96) / 25.4;
    expect(m.contentHeightPx).toBeCloseTo(m.heightPx - margins, 5);
    expect(m.pagination.contentHeight).toBeCloseTo(m.contentHeightPx - PAGE_SAFETY_PX, 5);
  });

  it('computes US Letter', () => {
    const m = pageMetrics('letter');
    expect(m.widthPx).toBeCloseTo(816, 1);
    expect(m.heightPx).toBeCloseTo(1056, 1);
  });

  it('falls back to A4 for an unknown size', () => {
    expect(pageMetrics('a3')).toEqual(pageMetrics('a4'));
  });
});

describe('pageCss', () => {
  it('emits an @page rule with the named size and no margin', () => {
    expect(pageCss('a4')).toBe('@page { size: A4; margin: 0; }');
    expect(pageCss('letter')).toBe('@page { size: Letter; margin: 0; }');
  });
});
