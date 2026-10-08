/**
 * Physical page geometry. Pages are drawn at CSS pixels (96 per inch), so the same numbers feed the
 * preview, the print stylesheet and the paginator's content height.
 */

export const MARGIN_MM = { top: 14, bottom: 14, x: 16 };

/**
 * Pagination fills a page to `contentHeight - PAGE_SAFETY_PX`, not to the last pixel, so that
 * sub-pixel differences between measuring and printing can never push a line off the page.
 */
export const PAGE_SAFETY_PX = 2;

const PX_PER_MM = 96 / 25.4;
const SIZES = {
  a4: { name: 'A4', widthMm: 210, heightMm: 297 },
  letter: { name: 'Letter', widthMm: 215.9, heightMm: 279.4 },
};

export function pageMetrics(size) {
  const id = SIZES[size] ? size : 'a4';
  const { name, widthMm, heightMm } = SIZES[id];
  const marginPx = {
    top: MARGIN_MM.top * PX_PER_MM,
    bottom: MARGIN_MM.bottom * PX_PER_MM,
    x: MARGIN_MM.x * PX_PER_MM,
  };
  const widthPx = widthMm * PX_PER_MM;
  const heightPx = heightMm * PX_PER_MM;
  const contentHeightPx = heightPx - marginPx.top - marginPx.bottom;
  return {
    size: id,
    name,
    widthMm,
    heightMm,
    widthPx,
    heightPx,
    marginPx,
    contentWidthPx: widthPx - 2 * marginPx.x,
    contentHeightPx,
    pagination: { contentHeight: contentHeightPx - PAGE_SAFETY_PX },
  };
}

/** The `@page` rule that makes the printed sheet match the preview page. */
export function pageCss(size) {
  return `@page { size: ${pageMetrics(size).name}; margin: 0; }`;
}
