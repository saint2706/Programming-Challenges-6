import { longResume } from '../src/lib/sample-long.js';
import { STORAGE_KEY, STORAGE_VERSION } from '../src/lib/storage.js';

/** The localStorage value the app reads on start, for a given document. */
export function storedValue(doc) {
  return JSON.stringify({ version: STORAGE_VERSION, savedAt: new Date().toISOString(), doc });
}

/** Open the app with `doc` already saved in the page's localStorage, and wait until it paginated. */
export async function openWith(browser, url, doc, { width = 1400, height = 1000 } = {}) {
  const page = await browser.newPage();
  await page.setViewport({ width, height, deviceScaleFactor: 1 });
  await page.evaluateOnNewDocument(
    (key, value) => {
      try {
        localStorage.setItem(key, value);
      } catch {
        // storage unavailable: the app starts from its default
      }
    },
    STORAGE_KEY,
    storedValue(doc),
  );
  // `?e2e` turns on the app's read-only test hook (`window.__resume`)
  await page.goto(`${url}/?e2e`, { waitUntil: 'load' });
  await page.waitForSelector('[data-ready="true"]', { timeout: 30_000 });
  await page.evaluate(() => document.fonts.ready);
  return page;
}

export { longResume };
