import { createRequire } from 'node:module';
import { decodeHash } from '../src/share.js';

const require = createRequire(import.meta.url);
export const AXE_PATH = require.resolve('axe-core/axe.min.js');

/** A page in its own browser context, so each test starts with empty localStorage. Closing the page closes the context. */
export async function freshPage(browser) {
  const context = await browser.createBrowserContext();
  const page = await context.newPage();
  page.close = () => context.close();
  return page;
}

/** Open the built app in a new page. Console errors and page errors are collected in `errors`. */
export async function openApp(browser, url, { path = '', hash = '', width = 1400, height = 1000, reduceMotion = false, dark = false, light = false, init } = {}) {
  const page = await freshPage(browser);
  await page.setViewport({ width, height });
  const errors = [];
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  page.on('pageerror', (e) => errors.push(e.message));
  const features = [];
  if (reduceMotion) features.push({ name: 'prefers-reduced-motion', value: 'reduce' });
  if (dark || light) features.push({ name: 'prefers-color-scheme', value: dark ? 'dark' : 'light' });
  if (features.length) await page.emulateMediaFeatures(features);
  if (init) await page.evaluateOnNewDocument(init);
  await page.goto(`${url}/${path}${hash}`);
  await page.waitForSelector('.pane');
  return { page, errors };
}

/** The colour (8-bit RGB) the browser actually paints for each token, per preview pane. */
export function paintedTokens(page, tokens) {
  return page.evaluate((list) => {
    const cx = Object.assign(document.createElement('canvas'), { width: 1, height: 1 }).getContext('2d', { willReadFrequently: true });
    const rgb = (color) => {
      cx.clearRect(0, 0, 1, 1);
      cx.fillStyle = '#000';
      cx.fillStyle = color;
      cx.fillRect(0, 0, 1, 1);
      return [...cx.getImageData(0, 0, 1, 1).data].slice(0, 3);
    };
    const out = {};
    for (const pane of document.querySelectorAll('.pane')) {
      const probe = document.createElement('i');
      pane.append(probe);
      out[pane.dataset.scheme] = {};
      for (const [key, css] of list) {
        probe.style.color = '';
        probe.style.color = `var(${css})`;
        out[pane.dataset.scheme][key] = rgb(getComputedStyle(probe).color);
      }
      probe.remove();
    }
    return out;
  }, tokens);
}

export const maxDelta = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));

/** The theme the page currently shows, read from its share link. */
export async function currentTheme(page) {
  const url = await page.$eval('#share-url', (e) => e.value);
  return decodeHash(new URL(url).hash).theme;
}

export async function runAxe(page, context) {
  await page.addScriptTag({ path: AXE_PATH });
  return page.evaluate(async (ctx) => {
    const r = await window.axe.run(ctx ? document.querySelector(ctx) : document);
    return r.violations.map((v) => ({ id: v.id, impact: v.impact, nodes: v.nodes.map((n) => n.target.join(' ')).slice(0, 5) }));
  }, context);
}

/** Set a numeric control through its exact-value box, the way a user types. */
export async function typeValue(page, key, value) {
  await page.focus(`#num-${key}`);
  await page.$eval(`#num-${key}`, (e) => e.select());
  await page.keyboard.type(String(value));
}
