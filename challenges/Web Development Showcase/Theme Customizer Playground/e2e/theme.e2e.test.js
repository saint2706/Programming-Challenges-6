import fc from 'fast-check';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { toRgb8 } from '../src/color.js';
import { wcagRatio } from '../src/contrast.js';
import { COLOR_TOKENS, deriveColors, ramp } from '../src/derive.js';
import { PRESETS } from '../src/presets.js';
import { defaults } from '../src/schema.js';
import { encodeHash } from '../src/share.js';
import { themeArb } from '../tests/helpers.js';
import { findChrome, launchChrome } from './chrome.mjs';
import { currentTheme, freshPage, maxDelta, openApp, paintedTokens, runAxe, typeValue } from './helpers.mjs';
import { startServer } from './server.mjs';

/**
 * The app in a real Chrome. The central check: the browser's own evaluation of the CSS derivation
 * (relative colours, color-mix, light-dark) agrees with the JS mirror that feeds the contrast table,
 * the exports and the fallback. Skipped (with a message) when no Chrome or Edge is installed.
 */

const chrome = findChrome();
if (!chrome) console.warn('[e2e] SKIPPED: no Chrome or Edge found. Set CHROME_PATH to run the browser tests.');

let server;
let browser;

beforeAll(async () => {
  if (!chrome) return;
  server = await startServer();
  browser = await launchChrome();
});

afterAll(async () => {
  await browser?.close();
  await server?.close();
});

const open = (options) => openApp(browser, server.url, options);
const TOKENS = [
  ...Object.entries(COLOR_TOKENS),
  ...[50, 100, 200, 300, 400, 500, 600, 700, 800, 900].map((step) => [`ramp${step}`, `--brand-${step}`]),
];

/** The JS mirror's colour for a token key. */
function expected(theme, scheme, key) {
  const color = key.startsWith('ramp') ? ramp(theme).find((r) => `ramp${r.step}` === key).color : deriveColors(theme, scheme)[key];
  return toRgb8(color);
}

async function worstDelta(options, themes) {
  let worst = { delta: 0 };
  for (const theme of themes) {
    const { page } = await open({ ...options, hash: encodeHash(theme) });
    const painted = await paintedTokens(page, TOKENS);
    for (const scheme of ['light', 'dark']) {
      for (const [key] of TOKENS) {
        const delta = maxDelta(painted[scheme][key], expected(theme, scheme, key));
        if (delta > worst.delta) worst = { delta, key, scheme, theme, painted: painted[scheme][key], expected: expected(theme, scheme, key) };
      }
    }
    await page.close();
  }
  return worst;
}

const themes = [defaults(), ...PRESETS.map((p) => p.theme), ...fc.sample(themeArb, { seed: 7, numRuns: 20 })];

describe.skipIf(!chrome)('loading', () => {
  it('shows the default theme in both modes with no console errors', async () => {
    const { page, errors } = await open({});
    expect(await page.$$eval('.pane', (p) => p.map((x) => x.dataset.scheme))).toEqual(['light', 'dark']);
    expect(await currentTheme(page)).toEqual(defaults());
    expect(await page.$eval('#fallback-banner', (e) => e.hidden)).toBe(true);
    expect(await page.$$eval('#contrast tbody tr', (r) => r.length)).toBe(16);
    expect(errors).toEqual([]);
    await page.close();
  });

  it('applies a theme from the link, and says so; a bad link is ignored with the reason', async () => {
    const good = await open({ hash: encodeHash({ ...defaults(), radius: 20, bodyFont: 'serif' }) });
    expect(await good.page.$eval('#num-radius', (e) => e.value)).toBe('20');
    expect(await good.page.$eval('#in-bodyFont', (e) => e.value)).toBe('serif');
    expect(await good.page.$eval('#notice', (e) => e.textContent)).toMatch(/from the link/);
    const bad = await open({ hash: '#radius=500&brandH=abc' });
    expect(await bad.page.$eval('#num-radius', (e) => e.value)).toBe('10');
    expect(await bad.page.$eval('#notice', (e) => e.textContent)).toMatch(/not used/);
    await good.page.close();
    await bad.page.close();
  });
});

describe.skipIf(!chrome)('the CSS derivation agrees with the JS mirror', () => {
  it('native CSS: every token, both modes, every theme in the list', async () => {
    const worst = await worstDelta({}, themes);
    console.log(`[e2e] native CSS vs JS: worst channel difference ${worst.delta} of 255`, worst.delta ? JSON.stringify(worst) : '');
    expect(worst.delta).toBeLessThanOrEqual(2);
  });

  it('JS fallback (?fallback): banner shown, and the same colours within one step', async () => {
    const { page } = await open({ path: '?fallback' });
    expect(await page.$eval('#fallback-banner', (e) => e.hidden)).toBe(false);
    await page.close();
    const worst = await worstDelta({ path: '?fallback' }, themes);
    console.log(`[e2e] JS fallback vs JS: worst channel difference ${worst.delta} of 255`);
    expect(worst.delta).toBe(0);
  });

  it('native and fallback paint the sample UI the same, element by element', async () => {
    const theme = PRESETS[2].theme;
    const read = async (path) => {
      const { page } = await open({ path, hash: encodeHash(theme) });
      const styles = await page.$$eval('.pane .s-btn, .pane .s-chip, .pane .s-alert, .pane .s-link, .pane .s-card', (els) =>
        els.map((e) => {
          const cx = Object.assign(document.createElement('canvas'), { width: 1, height: 1 }).getContext('2d', { willReadFrequently: true });
          const rgb = (c) => {
            cx.clearRect(0, 0, 1, 1);
            cx.fillStyle = '#000';
            cx.fillStyle = c;
            cx.fillRect(0, 0, 1, 1);
            return [...cx.getImageData(0, 0, 1, 1).data].slice(0, 3);
          };
          const cs = getComputedStyle(e);
          return { bg: rgb(cs.backgroundColor), fg: rgb(cs.color) };
        }),
      );
      await page.close();
      return styles;
    };
    const [native, fallback] = [await read(''), await read('?fallback')];
    expect(native.length).toBe(fallback.length);
    native.forEach((n, i) => {
      expect(maxDelta(n.bg, fallback[i].bg)).toBeLessThanOrEqual(2);
      expect(maxDelta(n.fg, fallback[i].fg)).toBeLessThanOrEqual(2);
    });
  });
});

describe.skipIf(!chrome)('accessibility (axe)', () => {
  it('has no violations with the default theme, light or dark chrome', async () => {
    for (const dark of [false, true]) {
      const { page } = await open({ dark, reduceMotion: true });
      expect(await runAxe(page), dark ? 'dark chrome' : 'light chrome').toEqual([]);
      await page.close();
    }
  });

  it.each(PRESETS.map((p) => p.name))('has no violations after applying the %s preset', async (name) => {
    const { page } = await open({ reduceMotion: true });
    await page.evaluate((n) => [...document.querySelectorAll('#presets button')].find((b) => b.textContent === n).click(), name);
    expect(await runAxe(page, '#panes')).toEqual([]);
    await page.close();
  });

  it('every control has a name and the page can be tabbed through to the end', async () => {
    const { page } = await open({});
    const seen = new Set();
    for (let i = 0; i < 120; i++) {
      await page.keyboard.press('Tab');
      const id = await page.evaluate(() => document.activeElement?.id || document.activeElement?.tagName);
      seen.add(id);
    }
    for (const id of ['in-brandL', 'num-brandL', 'in-bodyFont', 'target', 'export-text', 'import-file', 'import-apply', 'copy-link']) expect(seen.has(id), id).toBe(true);
    await page.close();
  });
});

describe.skipIf(!chrome)('editing', () => {
  it('arrow keys on a slider change the value, the box, the preview and the share link', async () => {
    const { page } = await open({});
    await page.focus('#in-brandH');
    for (let i = 0; i < 3; i++) await page.keyboard.press('ArrowRight');
    expect(await page.$eval('#num-brandH', (e) => e.value)).toBe('267');
    expect(await page.$eval('.pane', (p) => getComputedStyle(p).getPropertyValue('--brand-h').trim())).toBe('267');
    expect((await currentTheme(page)).brandH).toBe(267);
    expect(await page.evaluate(() => location.hash)).toContain('brandH=267');
    await page.close();
  });

  it('typing in the box edits live; an invalid value is flagged, not applied, and put right on blur', async () => {
    const { page } = await open({});
    await typeValue(page, 'radius', 24);
    expect(await page.$eval('.pane', (p) => getComputedStyle(p).getPropertyValue('--radius').trim())).toBe('24px');
    await typeValue(page, 'radius', 99);
    expect(await page.$eval('#num-radius', (e) => e.getAttribute('aria-invalid'))).not.toBeNull();
    expect((await currentTheme(page)).radius).toBe(9);
    await page.focus('#in-brandH');
    expect(await page.$eval('#num-radius', (e) => [e.value, e.hasAttribute('aria-invalid')])).toEqual(['9', false]);
    await page.close();
  });

  it('a brand colour from hex sets l, c and h; a bad hex is flagged', async () => {
    const { page } = await open({});
    await page.type('#brand-hex', '#2e7d32');
    await page.keyboard.press('Tab');
    const t = await currentTheme(page);
    expect(t.brandH).toBeGreaterThan(140);
    expect(t.brandH).toBeLessThan(150);
    await page.$eval('#brand-hex', (e) => (e.value = ''));
    await page.type('#brand-hex', 'nope');
    await page.keyboard.press('Tab');
    expect(await page.$eval('#brand-hex', (e) => e.getAttribute('aria-invalid'))).toBe('true');
    await page.close();
  });

  it('warns when the brand colour is outside sRGB', async () => {
    const { page } = await open({});
    await typeValue(page, 'brandC', 0.37);
    expect(await page.$eval('#gamut-note', (e) => [e.hidden, e.textContent])).toEqual([false, expect.stringContaining('outside sRGB')]);
    await page.close();
  });

  it('the Light / Dark / Both switch hides the other pane', async () => {
    const { page } = await open({});
    await page.click('input[name="view"][value="dark"]');
    expect(await page.$$eval('.pane', (p) => p.map((x) => x.hidden))).toEqual([true, false]);
    await page.click('input[name="view"][value="both"]');
    expect(await page.$$eval('.pane', (p) => p.map((x) => x.hidden))).toEqual([false, false]);
    await page.close();
  });

  it('presets apply and Reset undoes them; the URL follows', async () => {
    const { page } = await open({});
    await page.evaluate(() => [...document.querySelectorAll('#presets button')].find((b) => b.textContent === 'Forest').click());
    expect(await currentTheme(page)).toEqual(PRESETS[1].theme);
    expect(await page.evaluate(() => location.hash)).toContain('brandH=152');
    await page.click('#reset');
    expect(await currentTheme(page)).toEqual(defaults());
    await page.close();
  });

  it('does not scroll sideways on a phone-sized screen', async () => {
    const { page } = await open({ width: 375, height: 800 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    await page.close();
  });
});

describe.skipIf(!chrome)('contrast', () => {
  it('the table matches the ratio computed from the colours the browser paints', async () => {
    const theme = { ...defaults(), brandL: 0.7, brandH: 90 };
    const { page } = await open({ hash: encodeHash(theme) });
    const painted = await paintedTokens(page, TOKENS);
    for (const [scheme, id, fg, bg] of [
      ['light', 'brand-surface', 'brand', 'surface'],
      ['dark', 'text-bg', 'text', 'bg'],
      ['light', 'onbrand-brand', 'onBrand', 'brand'],
    ]) {
      const shown = await page.$eval(`tr[data-pair="${id}"][data-scheme="${scheme}"] td:nth-child(4)`, (e) => e.textContent);
      expect(Number.parseFloat(shown), `${scheme} ${id}`).toBeCloseTo(wcagRatio(painted[scheme][fg], painted[scheme][bg]), 1);
    }
    await page.close();
  });

  it('a failing pair offers the nearest passing lightness, and applying it fixes the pair', async () => {
    const { page } = await open({ reduceMotion: true });
    await typeValue(page, 'brandL', 0.75);
    const row = 'tr[data-pair="brand-surface"][data-scheme="light"]';
    expect(await page.$eval(row, (r) => r.dataset.pass)).toBe('false');
    const offered = await page.$eval(`${row} .fix`, (b) => b.textContent.replace('Set to ', ''));
    await page.click(`${row} .fix`);
    expect(await page.$eval('#num-brandL', (e) => e.value)).toBe(offered);
    expect(await page.$eval(row, (r) => r.dataset.pass)).toBe('true');
    expect(Number.parseFloat(await page.$eval(`${row} td:nth-child(4)`, (e) => e.textContent))).toBeGreaterThanOrEqual(4.5);
    expect(await page.evaluate(() => document.activeElement.id)).toBe('contrast-region');
    await page.close();
  });

  it('the fix target changes what passes: APCA Lc 75 flags the default, and offers fixes', async () => {
    const { page } = await open({});
    await page.select('#target', 'apca75');
    const failing = await page.$$eval('#contrast tbody tr', (rows) => rows.filter((r) => r.dataset.pass === 'false').length);
    expect(failing).toBeGreaterThan(0);
    await page.select('#target', 'aa');
    expect(await page.$$eval('#contrast tbody tr', (rows) => rows.every((r) => r.dataset.pass === 'true'))).toBe(true);
    await page.close();
  });
});

describe.skipIf(!chrome)('export and import', () => {
  const exportText = (page) => page.$eval('#export-text', (e) => e.value);
  const choose = (page, id) => page.click(`input[name="format"][value="${id}"]`);
  const importPasted = async (page, text) => {
    await page.$eval('#import-text', (e, v) => (e.value = v), text);
    await page.click('#import-apply');
  };

  it.each(['css', 'tokens'])('export (%s), reset, import: the same theme comes back', async (format) => {
    const { page } = await open({ reduceMotion: true });
    await page.evaluate(() => [...document.querySelectorAll('#presets button')].find((b) => b.textContent === 'Sunset').click());
    const before = await currentTheme(page);
    await choose(page, format);
    const text = await exportText(page);
    await page.click('#reset');
    expect(await currentTheme(page)).toEqual(defaults());
    await importPasted(page, text);
    expect(await page.$eval('#import-result', (e) => e.textContent)).toMatch(/Imported/);
    expect(await currentTheme(page)).toEqual(before);
    await page.close();
  });

  it('the exports have the expected shape in the page', async () => {
    const { page } = await open({});
    expect(await exportText(page)).toMatch(/^\/\*.*\n:root \{\n {2}--brand-l: 0\.55;/);
    await choose(page, 'tokens');
    expect(JSON.parse(await exportText(page)).theme.color['brand-light'].$value.colorSpace).toBe('oklch');
    await choose(page, 'tailwind');
    expect(await exportText(page)).toMatch(/@theme \{\n {2}--color-brand-50: oklch\(/);
    await page.close();
  });

  it('the exported CSS, dropped into a bare page, reproduces the painted colours', async () => {
    const theme = PRESETS[1].theme;
    const { page } = await open({ hash: encodeHash(theme) });
    const css = await exportText(page);
    const painted = await paintedTokens(page, TOKENS);
    const bare = await freshPage(browser);
    await bare.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'light' }]);
    await bare.setContent(`<!doctype html><style>${css}</style><body><section class="pane" data-scheme="light"></section></body>`);
    const bareTokens = await bare.evaluate((list) => {
      const cx = Object.assign(document.createElement('canvas'), { width: 1, height: 1 }).getContext('2d', { willReadFrequently: true });
      const probe = document.createElement('i');
      document.body.append(probe);
      return Object.fromEntries(
        list.map(([key, css]) => {
          probe.style.color = `var(${css})`;
          cx.clearRect(0, 0, 1, 1);
          cx.fillStyle = '#000';
          cx.fillStyle = getComputedStyle(probe).color;
          cx.fillRect(0, 0, 1, 1);
          return [key, [...cx.getImageData(0, 0, 1, 1).data].slice(0, 3)];
        }),
      );
    }, TOKENS);
    for (const [key] of TOKENS) expect(maxDelta(bareTokens[key], painted.light[key]), key).toBeLessThanOrEqual(1);
    await bare.close();
    await page.close();
  });

  it('Download hands out exactly the text on screen, under a fixed file name', async () => {
    const { page } = await open({});
    await page.evaluate(() => {
      const make = URL.createObjectURL;
      URL.createObjectURL = (blob) => ((window.__blob = blob), make(blob));
      HTMLAnchorElement.prototype.click = function () {
        window.__download = this.download;
      };
    });
    await choose(page, 'tokens');
    await page.click('#download');
    expect(await page.evaluate(() => window.__download)).toBe('theme.tokens.json');
    expect(await page.evaluate(() => window.__blob.text())).toBe(await exportText(page));
    await page.close();
  });

  it('a share link opened in a fresh page shows the same theme', async () => {
    const first = await open({});
    await typeValue(first.page, 'radius', 18);
    const link = await first.page.$eval('#share-url', (e) => e.value);
    const second = await freshPage(browser);
    await second.goto(link);
    expect(await second.$eval('#num-radius', (e) => e.value)).toBe('18');
    await first.page.close();
    await second.close();
  });

  it('a file upload imports; an oversized file is refused', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'theme-e2e-'));
    const good = join(dir, 'brand.css');
    writeFileSync(good, ':root { --brand-h: 120; --radius: 3px; }');
    const big = join(dir, 'big.css');
    writeFileSync(big, `:root{--radius:1px}${' '.repeat(150_000)}`);
    const { page } = await open({});
    await (await page.$('#import-file')).uploadFile(good);
    await page.waitForFunction(() => /Imported/.test(document.querySelector('#import-result').textContent));
    expect((await currentTheme(page)).brandH).toBe(120);
    await (await page.$('#import-file')).uploadFile(big);
    await page.waitForFunction(() => /larger than/.test(document.querySelector('#import-result').textContent));
    expect((await currentTheme(page)).radius).toBe(3);
    await page.close();
  });

  it('hostile files change nothing and inject nothing', async () => {
    const { page, errors } = await open({});
    const before = { theme: await currentTheme(page), styles: await page.$$eval('style, link[rel="stylesheet"]', (n) => n.length), scripts: await page.$$eval('script', (n) => n.length) };
    const hostile = [
      ':root { --brand-c: 0.1; } --radius: 5px; } body { display: none !important } .pane { --radius: url(javascript:alert(1)) ',
      ':root { --radius: 5px; --brand-h: 9999 }',
      '@import url(https://evil.example/x.css); :root { --font-body: Papyrus, </style><script>window.pwned=1</script> }',
      JSON.stringify({ theme: { radius: { $type: 'dimension', $value: { value: '5px; } body { display:none', unit: 'px' } } } }),
      JSON.stringify({ theme: { typography: { 'body-font': { $value: ['x</style><script>window.pwned=1</script>'] } } } }),
    ];
    for (const text of hostile) {
      await page.$eval('#import-text', (e, v) => (e.value = v), text);
      await page.click('#import-apply');
      expect(await page.$eval('#import-result', (e) => e.textContent), text).toMatch(/Nothing was changed|Could not import/);
    }
    expect(await currentTheme(page)).toEqual(before.theme);
    expect(await page.$$eval('style, link[rel="stylesheet"]', (n) => n.length)).toBe(before.styles);
    expect(await page.$$eval('script', (n) => n.length)).toBe(before.scripts);
    expect(await page.evaluate(() => [typeof window.pwned, getComputedStyle(document.body).display])).toEqual(['undefined', 'block']);
    expect(errors).toEqual([]);
    await page.close();
  });

  it('a partly valid hostile CSS applies only the values it understood', async () => {
    const { page } = await open({});
    await page.$eval('#import-text', (e) => (e.value = 'body { display: none } :root { --radius: 5px; color: red; --brand-h: 100; } .x { background: url(x) }'));
    await page.click('#import-apply');
    expect(await currentTheme(page)).toEqual({ ...defaults(), radius: 5, brandH: 100 });
    expect(await page.evaluate(() => getComputedStyle(document.body).display)).toBe('block');
    await page.close();
  });
});

describe.skipIf(!chrome)('autosave', () => {
  it('keeps a change across a visit to the plain address', async () => {
    const { page } = await open({});
    await typeValue(page, 'space', 6);
    const saved = await page.evaluate(() => localStorage.getItem('theme-playground:v1'));
    expect(JSON.parse(saved).theme.space).toBe(6);
    await page.goto(server.url);
    expect(await page.$eval('#num-space', (e) => e.value)).toBe('6');
    await page.close();
  });

  it('corrupt saved data is set aside with a notice and the defaults load', async () => {
    const { page, errors } = await open({ init: () => localStorage.setItem('theme-playground:v1', '{"v":1,"theme":{"radius":"huge"}}') });
    expect(await page.$eval('#notice', (e) => e.textContent)).toMatch(/set aside/);
    expect(await currentTheme(page)).toEqual(defaults());
    expect(await page.evaluate(() => localStorage.getItem('theme-playground:backup'))).toContain('huge');
    expect(errors).toEqual([]);
    await page.close();
  });

  it('works, with a notice, when the browser blocks storage', async () => {
    const { page, errors } = await open({
      init: () => {
        Object.defineProperty(window, 'localStorage', {
          get() {
            throw new DOMException('blocked', 'SecurityError');
          },
        });
      },
    });
    expect(await page.$eval('#notice', (e) => e.textContent)).toMatch(/Saving is off/);
    await typeValue(page, 'radius', 7);
    expect((await currentTheme(page)).radius).toBe(7);
    expect(errors).toEqual([]);
    await page.close();
  });
});

describe.skipIf(!chrome)('motion', () => {
  const clickSunset = (page) => page.evaluate(() => [...document.querySelectorAll('#presets button')].find((b) => b.textContent === 'Sunset').click());
  const hue = (page) => page.$eval('.pane', (p) => Number.parseFloat(getComputedStyle(p).getPropertyValue('--brand-h')));

  it('a preset glides: the typed hue is between the old and new value mid-transition, then lands', async () => {
    const { page } = await open({});
    expect(await hue(page)).toBe(264);
    await clickSunset(page);
    await new Promise((r) => setTimeout(r, 120));
    const mid = await hue(page);
    console.log(`[e2e] --brand-h 120 ms into a 400 ms transition from 264 to 40: ${mid.toFixed(1)}`);
    expect(mid).toBeLessThan(264);
    expect(mid).toBeGreaterThan(40);
    await page.waitForFunction(() => !document.querySelector('.pane').classList.contains('animating'));
    expect(await hue(page)).toBe(40);
    await page.close();
  });

  it('with reduced motion it jumps at once', async () => {
    const { page } = await open({ reduceMotion: true });
    await clickSunset(page);
    expect(await hue(page)).toBe(40);
    expect(await page.$eval('.pane', (p) => getComputedStyle(p).transitionDuration)).toBe('0s');
    await page.close();
  });

  it('slider drags never animate (they would lag behind the pointer)', async () => {
    const { page } = await open({});
    await page.focus('#in-brandH');
    await page.keyboard.press('ArrowRight');
    expect(await page.$eval('.pane', (p) => p.classList.contains('animating'))).toBe(false);
    await page.close();
  });
});
