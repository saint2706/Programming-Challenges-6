import sharp from 'sharp';

/** Flat-colour SVGs: "before" is red, "after" is blue, so a screenshot says which side is showing. */
export const svg = (fill, w = 400, h = 200) =>
  `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}"><rect width="100%" height="100%" fill="${fill}"/></svg>`)}`;
export const RED = svg('#ff0000');
export const BLUE = svg('#0000ff');

export const SLIDER = (attrs = '') => `<img-compare id="s" before-src="${RED}" after-src="${BLUE}" before-alt="red" after-alt="blue" ${attrs}></img-compare>`;

/** Open the harness, put `html` in the body, and wait until both images of every slider are in. */
export async function openWith(browser, base, html, { width = 800, height = 700, hasTouch = false, waitReady = true } = {}) {
  const page = await browser.newPage();
  await page.setViewport({ width, height, hasTouch, isMobile: false });
  await page.goto(`${base}/e2e/harness.html`);
  await page.waitForFunction('window.harnessReady === true');
  await page.evaluate((markup) => {
    document.body.insertAdjacentHTML('beforeend', markup);
  }, html);
  if (waitReady) {
    await page.waitForFunction(() => [...document.querySelectorAll('img-compare')].every((el) => el.status !== 'loading'));
  }
  return page;
}

/** The slider's box in page coordinates. */
export const boxOf = (page, selector = '#s') =>
  page.evaluate((sel) => {
    const r = document.querySelector(sel).getBoundingClientRect();
    return { x: r.x + scrollX, y: r.y + scrollY, width: r.width, height: r.height };
  }, selector);

export const positionOf = (page, selector = '#s') => page.evaluate((sel) => document.querySelector(sel).position, selector);

/** Screenshot the slider and return a sampler: `at(xFraction, yFraction)` -> [r, g, b]. */
export async function sampler(page, selector = '#s') {
  const box = await boxOf(page, selector);
  const png = await page.screenshot({ clip: { x: box.x, y: box.y, width: box.width, height: box.height } });
  const { data, info } = await sharp(png).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  return (fx, fy) => {
    const x = Math.min(info.width - 1, Math.round(fx * info.width));
    const y = Math.min(info.height - 1, Math.round(fy * info.height));
    const i = (y * info.width + x) * info.channels;
    return [data[i], data[i + 1], data[i + 2]];
  };
}

export const isRed = ([r, g, b]) => r > 230 && g < 30 && b < 30;
export const isBlue = ([r, g, b]) => b > 230 && r < 30 && g < 30;
