import { createRequire } from 'node:module';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { findChrome, launchChrome } from './chrome.mjs';
import { BLUE, RED, SLIDER, boxOf, isBlue, isRed, openWith, positionOf, sampler, svg } from './fixtures.mjs';
import { startServer } from './server.mjs';

/**
 * The component in a real Chrome or Edge, against the built bundle (`dist/`): pointer capture,
 * touch gestures, painted pixels, RTL, lazy loading, reduced motion and axe. Skipped (with a
 * message) when no browser is installed.
 */

const chrome = findChrome();
if (!chrome) console.warn('[e2e] SKIPPED: no Chrome or Edge found. Set CHROME_PATH to run the browser tests.');

let server;
let browser;
const require = createRequire(import.meta.url);

beforeAll(async () => {
  if (!chrome) return;
  server = await startServer();
  browser = await launchChrome();
});

afterAll(async () => {
  await browser?.close();
  await server?.close();
});

const open = (html, options) => openWith(browser, server.url, html, options);
const settle = () => new Promise((resolve) => setTimeout(resolve, 300)); // longer than the 140 ms transition
const near = (actual, expected, tolerance = 0.6) => expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tolerance);

/** Collect input/change events from the first slider into window.__log. */
const logEvents = (page) =>
  page.evaluate(() => {
    window.__log = [];
    for (const type of ['input', 'change'])
      document.querySelector('#s').addEventListener(type, (e) => window.__log.push([type, e.detail.position]));
  });
const eventLog = (page) => page.evaluate(() => window.__log);

describe.skipIf(!chrome)('what is painted', () => {
  it('shows the before image on the start side of the divider and the after image beyond it', async () => {
    const page = await open(SLIDER('position="30"'));
    const at = await sampler(page);
    expect(isRed(at(0.1, 0.8))).toBe(true);
    expect(isRed(at(0.26, 0.8))).toBe(true);
    expect(isBlue(at(0.34, 0.8))).toBe(true);
    expect(isBlue(at(0.9, 0.8))).toBe(true);
    await page.close();
  });

  it('is all after at 0 and all before at 100', async () => {
    const page = await open(SLIDER('position="0"'));
    expect(isBlue((await sampler(page))(0.05, 0.8))).toBe(true);
    await page.evaluate(() => (document.querySelector('#s').position = 100));
    await settle();
    expect(isRed((await sampler(page))(0.95, 0.8))).toBe(true);
    await page.close();
  });

  it('mirrors under dir=rtl: the before image starts at the right edge', async () => {
    const page = await open(SLIDER('position="30"'));
    await page.evaluate(() => (document.documentElement.dir = 'rtl'));
    await settle();
    const at = await sampler(page);
    expect(isRed(at(0.9, 0.8))).toBe(true);
    expect(isRed(at(0.76, 0.8))).toBe(true);
    expect(isBlue(at(0.64, 0.8))).toBe(true);
    expect(isBlue(at(0.1, 0.8))).toBe(true);
    await page.close();
  });

  it('vertical: the before image fills from the top', async () => {
    const page = await open(SLIDER('position="30" orientation="vertical"'));
    const at = await sampler(page);
    expect(isRed(at(0.5, 0.1))).toBe(true); // x = 0.5 keeps clear of the corner label
    expect(isRed(at(0.5, 0.2))).toBe(true);
    expect(isBlue(at(0.5, 0.45))).toBe(true);
    expect(isBlue(at(0.9, 0.9))).toBe(true);
    await page.close();
  });

  it('clips slotted content too (anything in the before/after slots, not only images)', async () => {
    const page = await open(
      `<img-compare id="s" position="40" ratio="2/1"><div slot="before" style="background:#f00"></div><div slot="after" style="background:#00f"></div></img-compare>`,
    );
    const at = await sampler(page);
    expect(isRed(at(0.1, 0.8))).toBe(true);
    expect(isBlue(at(0.7, 0.8))).toBe(true);
    await page.close();
  });

  it('keeps the divider line on the boundary (within a pixel) at several positions', async () => {
    const page = await open(SLIDER());
    for (const p of [10, 33, 50, 77, 90]) {
      const offset = await page.evaluate(async (value) => {
        const el = document.querySelector('#s');
        el.position = value;
        await new Promise((resolve) => setTimeout(resolve, 300));
        const track = el.shadowRoot.querySelector('.line').getBoundingClientRect();
        return track.left + track.width / 2 - (el.getBoundingClientRect().left + (el.getBoundingClientRect().width * value) / 100);
      }, p);
      near(offset, 0, 1.01);
    }
    await page.close();
  });
});

describe.skipIf(!chrome)('mouse', () => {
  it('press jumps, drag follows, pointer capture keeps tracking outside the box, release emits one change', async () => {
    const page = await open(SLIDER());
    await logEvents(page);
    const box = await boxOf(page);
    const y = box.y + box.height / 2;
    await page.mouse.move(box.x + box.width * 0.25, y);
    await page.mouse.down();
    near(await positionOf(page), 25);
    await page.mouse.move(box.x + box.width * 0.75, y, { steps: 5 });
    near(await positionOf(page), 75);
    await page.mouse.move(box.x + box.width + 250, box.y + box.height + 150, { steps: 4 }); // far outside
    expect(await positionOf(page)).toBe(100);
    await page.mouse.move(box.x - 250, box.y - 10, { steps: 4 });
    expect(await positionOf(page)).toBe(0);
    await page.mouse.up();
    const log = await eventLog(page);
    expect(log.filter(([type]) => type === 'change')).toHaveLength(1);
    expect(log.at(-1)).toEqual(['change', 0]);
    expect(log.filter(([type]) => type === 'input').length).toBeGreaterThan(4);
    // After release, moving does nothing.
    await page.mouse.move(box.x + box.width * 0.6, y);
    expect(await positionOf(page)).toBe(0);
    await page.close();
  });

  it('dragging does not select page text or start a native image drag', async () => {
    const page = await open(`<p id="t">Some text around the slider</p>${SLIDER()}<p>More text below</p>`);
    const box = await boxOf(page);
    await page.mouse.move(box.x + 100, box.y + box.height * 0.8);
    await page.mouse.down();
    await page.mouse.move(box.x + 400, box.y + box.height * 0.8, { steps: 6 });
    await page.mouse.up();
    expect(await page.evaluate(() => String(getSelection()))).toBe('');
    await page.close();
  });

  it('ignores the right mouse button', async () => {
    const page = await open(SLIDER('position="50"'));
    const box = await boxOf(page);
    await page.mouse.click(box.x + 60, box.y + box.height * 0.8, { button: 'right' });
    expect(await positionOf(page)).toBe(50);
    await page.close();
  });

  it('vertical orientation reads the y axis', async () => {
    const page = await open(SLIDER('orientation="vertical"'));
    const box = await boxOf(page);
    await page.mouse.move(box.x + 100, box.y + box.height * 0.2);
    await page.mouse.down();
    near(await positionOf(page), 20);
    await page.mouse.move(box.x + 500, box.y + box.height * 0.9, { steps: 4 });
    near(await positionOf(page), 90);
    await page.mouse.up();
    await page.close();
  });

  it('rtl measures from the right edge', async () => {
    const page = await open(SLIDER());
    await page.evaluate(() => (document.documentElement.dir = 'rtl'));
    const box = await boxOf(page);
    await page.mouse.move(box.x + box.width * 0.8, box.y + box.height * 0.8);
    await page.mouse.down();
    near(await positionOf(page), 20);
    await page.mouse.up();
    await page.close();
  });

  it('hover mode follows an unpressed mouse and emits change when it leaves', async () => {
    const page = await open(SLIDER('hover'));
    await logEvents(page);
    const box = await boxOf(page);
    await page.mouse.move(box.x + box.width * 0.2, box.y + box.height * 0.8);
    await page.mouse.move(box.x + box.width * 0.7, box.y + box.height * 0.8, { steps: 5 });
    near(await positionOf(page), 70);
    await page.mouse.move(box.x + box.width + 100, box.y + box.height * 0.8);
    const log = await eventLog(page);
    expect(log.at(-1)[0]).toBe('change');
    near(log.at(-1)[1], 70, 15); // the last position over the box
    await page.close();
  });

  it('a hover-mode slider does not follow the mouse when the attribute is removed', async () => {
    const page = await open(SLIDER('hover position="50"'));
    await page.evaluate(() => document.querySelector('#s').removeAttribute('hover'));
    const box = await boxOf(page);
    await page.mouse.move(box.x + box.width * 0.2, box.y + box.height * 0.8);
    await page.mouse.move(box.x + box.width * 0.1, box.y + box.height * 0.8, { steps: 3 });
    expect(await positionOf(page)).toBe(50);
    await page.close();
  });
});

describe.skipIf(!chrome)('keyboard', () => {
  it('Tab reaches the handle (one tab stop), arrows/Shift/Home/End/Page keys move it, Tab leaves', async () => {
    const page = await open(`<button id="before">before</button>${SLIDER('position="50"')}<button id="after">after</button>`);
    await page.focus('#before');
    await page.keyboard.press('Tab');
    const focus = await page.evaluate(() => {
      const el = document.querySelector('#s');
      return { host: document.activeElement === el, inner: el.shadowRoot.activeElement?.getAttribute('role') };
    });
    expect(focus).toEqual({ host: true, inner: 'slider' });
    await logEvents(page);
    for (let i = 0; i < 3; i++) await page.keyboard.press('ArrowRight');
    expect(await positionOf(page)).toBe(53);
    await page.keyboard.down('Shift');
    await page.keyboard.press('ArrowLeft');
    await page.keyboard.up('Shift');
    expect(await positionOf(page)).toBe(43);
    await page.keyboard.press('PageUp');
    expect(await positionOf(page)).toBe(53);
    await page.keyboard.press('End');
    expect(await positionOf(page)).toBe(100);
    await page.keyboard.press('Home');
    expect(await positionOf(page)).toBe(0);
    const log = await eventLog(page);
    expect(log.filter(([t]) => t === 'change')).toHaveLength(log.filter(([t]) => t === 'input').length);
    await page.keyboard.press('Tab');
    expect(await page.evaluate(() => document.activeElement.id)).toBe('after');
    await page.close();
  });

  it('arrow keys do not scroll the page while the slider has focus', async () => {
    const page = await open(`<div style="height:2500px"></div>${SLIDER()}<div style="height:2500px"></div>`, { height: 500, waitReady: false });
    await page.evaluate(() => document.querySelector('#s').shadowRoot.querySelector('.handle').focus());
    const before = await page.evaluate(() => scrollY);
    await page.keyboard.press('ArrowDown');
    await page.keyboard.press('PageDown');
    await page.keyboard.press('Home');
    expect(await page.evaluate(() => scrollY)).toBe(before);
    await page.close();
  });

  it('shows a focus ring on keyboard focus only', async () => {
    const page = await open(SLIDER());
    await page.keyboard.press('Tab');
    const outline = () =>
      page.evaluate(() => getComputedStyle(document.querySelector('#s').shadowRoot.querySelector('.knob')).outlineStyle);
    expect(await outline()).toBe('solid');
    await page.close();
  });

  it('rtl flips Left/Right, vertical uses Down for more', async () => {
    const page = await open(SLIDER());
    await page.evaluate(() => (document.documentElement.dir = 'rtl'));
    await page.keyboard.press('Tab');
    await page.keyboard.press('ArrowRight');
    expect(await positionOf(page)).toBe(49);
    await page.evaluate(() => {
      document.documentElement.dir = 'ltr';
      document.querySelector('#s').orientation = 'vertical';
    });
    await page.keyboard.press('ArrowDown');
    expect(await positionOf(page)).toBe(50);
    await page.close();
  });
});

describe.skipIf(!chrome)('touch', () => {
  const swipe = async (page, from, to, steps = 8) => {
    const touch = await page.touchscreen.touchStart(from.x, from.y);
    for (let i = 1; i <= steps; i++) {
      await touch.move(from.x + ((to.x - from.x) * i) / steps, from.y + ((to.y - from.y) * i) / steps);
    }
    await touch.end();
  };

  it('a horizontal swipe that starts on the image drags the divider', async () => {
    const page = await open(`<div style="height:300px"></div>${SLIDER('position="50"')}<div style="height:1800px"></div>`, { hasTouch: true });
    await page.evaluate(() => scrollTo(0, 250));
    const box = await boxOf(page);
    const y = box.y + box.height * 0.8 - 250;
    await swipe(page, { x: box.x + box.width * 0.2, y }, { x: box.x + box.width * 0.8, y });
    near(await positionOf(page), 80, 3);
    await page.close();
  });

  it('a vertical swipe over the slider scrolls the page and leaves the divider alone', async () => {
    const page = await open(`<div style="height:300px"></div>${SLIDER('position="50"')}<div style="height:3000px"></div>`, { hasTouch: true });
    await page.evaluate(() => scrollTo(0, 250));
    await logEvents(page);
    const box = await boxOf(page);
    const x = box.x + box.width * 0.3;
    const y = box.y + box.height * 0.8 - 250;
    const scrolled = await page.evaluate(() => scrollY);
    await swipe(page, { x, y }, { x: x + 4, y: y - 200 }, 10);
    await new Promise((r) => setTimeout(r, 300));
    expect(await page.evaluate(() => scrollY)).toBeGreaterThan(scrolled + 50);
    expect(await positionOf(page)).toBe(50);
    expect(await eventLog(page)).toEqual([]);
    await page.close();
  });

  it('a tap jumps there, and touching the handle drags immediately', async () => {
    const page = await open(SLIDER('position="50"'), { hasTouch: true });
    const box = await boxOf(page);
    const y = box.y + box.height * 0.8;
    await page.touchscreen.tap(box.x + box.width * 0.2, y);
    near(await positionOf(page), 20, 1);
    await settle(); // the divider glides to the tap; grab it once it has arrived
    const knob = await page.evaluate(() => {
      const r = document.querySelector('#s').shadowRoot.querySelector('.knob').getBoundingClientRect();
      return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
    });
    await swipe(page, knob, { x: knob.x + 150, y: knob.y + 60 }, 6); // diagonal is fine from the handle
    near(await positionOf(page), 20 + (150 / box.width) * 100, 3);
    await page.close();
  });

  it('on a vertical slider a horizontal swipe scrolls/does nothing, a vertical swipe drags', async () => {
    const page = await open(`${SLIDER('orientation="vertical" position="50"')}<div style="height:2000px"></div>`, { hasTouch: true });
    const box = await boxOf(page);
    const x = box.x + box.width * 0.3;
    await swipe(page, { x, y: box.y + box.height * 0.3 }, { x, y: box.y + box.height * 0.8 });
    near(await positionOf(page), 80, 3);
    await page.close();
  });
});

describe.skipIf(!chrome)('size, loading and errors', () => {
  it('takes its height from the image ratio, and a declared ratio stops layout shift while loading', async () => {
    const slow = `/slow.svg?ms=500&w=600&h=200`; // 3:1
    const withRatio = `<img-compare id="a" ratio="3/1" before-src="${slow}" after-src="${slow}"></img-compare>`;
    const without = `<img-compare id="b" before-src="${slow}&x=1" after-src="${slow}&x=1"></img-compare>`;
    const page = await open(withRatio + without, { waitReady: false });
    const heights = () => page.evaluate(() => ['a', 'b'].map((id) => document.getElementById(id).getBoundingClientRect().height));
    const early = await heights();
    await page.waitForFunction(() => [...document.querySelectorAll('img-compare')].every((el) => el.status === 'ready'));
    const late = await heights();
    near(early[0], 200, 0.5); // 600 wide / 3
    near(late[0], 200, 0.5); // unchanged: no shift
    near(early[1], 600 / (16 / 9), 0.5); // the 16/9 placeholder
    near(late[1], 200, 0.5); // jumped to the real ratio: the shift `ratio` avoids
    await page.close();
  });

  it('when the two images differ in shape, the first loaded (after) sets the box and both fill it', async () => {
    const tall = svg('#ff0000', 100, 100);
    const wide = svg('#0000ff', 400, 200);
    const page = await open(`<img-compare id="s" position="50" before-src="${tall}" after-src="${wide}"></img-compare>`);
    near((await boxOf(page)).height, 300, 0.5);
    const at = await sampler(page);
    expect(isRed(at(0.2, 0.8))).toBe(true); // `cover` scaled the square to fill the box
    expect(isBlue(at(0.8, 0.8))).toBe(true);
    await page.close();
  });

  it('fit="contain" letterboxes instead of cropping', async () => {
    const square = svg('#ff0000', 100, 100);
    const page = await open(`<img-compare id="s" position="100" ratio="2/1" fit="contain" before-src="${square}" after-src="${BLUE}"></img-compare>`);
    const box = await boxOf(page);
    const fit = await page.evaluate(() => getComputedStyle(document.querySelector('#s').shadowRoot.querySelector('.before img')).objectFit);
    expect(fit).toBe('contain');
    expect(box.width / box.height).toBeCloseTo(2, 1);
    await page.close();
  });

  it('shows an error overlay, sets status and emits image-error for a 404, without breaking the other image', async () => {
    const page = await open(
      `<img-compare id="s" before-src="${RED}" after-src="/missing.png"></img-compare>`,
      { waitReady: false },
    );
    await page.waitForFunction(() => document.querySelector('#s').status === 'error');
    const state = await page.evaluate(() => {
      const el = document.querySelector('#s');
      const error = el.shadowRoot.querySelector('.error');
      return { attr: el.getAttribute('data-status'), display: getComputedStyle(error).display, role: error.getAttribute('role'), text: error.textContent.trim() };
    });
    expect(state).toEqual({ attr: 'error', display: 'grid', role: 'alert', text: 'Image failed to load' });
    await page.close();
  });

  it('refuses a javascript: source without running it', async () => {
    const page = await open(`<img-compare id="s" before-src="javascript:window.__pwned=1" after-src="${BLUE}"></img-compare>`, { waitReady: false });
    await page.waitForFunction(() => document.querySelector('#s').status === 'error');
    expect(await page.evaluate(() => window.__pwned)).toBeUndefined();
    await page.close();
  });

  it('does not fetch images far below the fold until the slider nears the viewport (loading=lazy inside the shadow root)', async () => {
    const lazy = `/slow.svg?ms=0&w=400&h=200&x=lazy`;
    const page = await open(`<div style="height:9000px"></div><img-compare id="s" before-src="${lazy}1" after-src="${lazy}2"></img-compare>`, {
      waitReady: false,
    });
    await new Promise((r) => setTimeout(r, 500));
    const fetched = () => server.requests.filter((r) => r.includes('x=lazy')).length;
    expect(fetched()).toBe(0);
    await page.evaluate(() => document.querySelector('#s').scrollIntoView());
    await page.waitForFunction(() => document.querySelector('#s').status === 'ready');
    expect(fetched()).toBe(2);
    await page.close();
  });

  it('a hostile alt/label renders as text, never as markup', async () => {
    const hostile = '"><img src=x onerror="window.__pwned=1"><b>bold</b>';
    const page = await open(`<img-compare id="s" before-src="${RED}" after-src="${BLUE}" before-alt='${hostile.replace(/'/g, '&#39;')}' before-label='${hostile.replace(/'/g, '&#39;')}'></img-compare>`);
    const result = await page.evaluate(() => {
      const el = document.querySelector('#s');
      return {
        images: el.shadowRoot.querySelectorAll('img').length,
        bold: el.shadowRoot.querySelectorAll('b').length,
        label: el.shadowRoot.querySelector('.tag-before').textContent,
        pwned: window.__pwned,
      };
    });
    expect(result.images).toBe(2);
    expect(result.bold).toBe(0);
    expect(result.label).toContain('<b>bold</b>');
    expect(result.pwned).toBeUndefined();
    await page.close();
  });
});

describe.skipIf(!chrome)('motion and theming', () => {
  const durations = (page) =>
    page.evaluate(() => {
      const root = document.querySelector('#s').shadowRoot;
      return ['.before', '.track'].map((sel) => getComputedStyle(root.querySelector(sel)).transitionDuration);
    });

  it('animates jumps by default, never while dragging, and not at all under prefers-reduced-motion', async () => {
    const page = await open(SLIDER());
    expect(await durations(page)).toEqual(['0.14s', '0.14s']);
    const box = await boxOf(page);
    await page.mouse.move(box.x + 100, box.y + box.height * 0.8);
    await page.mouse.down();
    expect(await durations(page)).toEqual(['0s', '0s']);
    await page.mouse.up();
    await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }]);
    expect(await durations(page)).toEqual(['0s', '0s']);
    await page.close();
  });

  it('hover mode never transitions (the divider must stick to the mouse)', async () => {
    const page = await open(SLIDER('hover'));
    expect(await durations(page)).toEqual(['0s', '0s']);
    await page.close();
  });

  it('custom properties, ::part() and slots restyle the handle and labels', async () => {
    const page = await open(`
      <style>
        #s { --ic-line-color: rgb(255, 128, 0); --ic-line-width: 6px; --ic-knob-size: 60px; }
        #s::part(knob) { background: rgb(0, 128, 0); }
        #s::part(label) { text-transform: uppercase; }
      </style>
      <img-compare id="s" before-src="${RED}" after-src="${BLUE}" before-label="gray">
        <span slot="handle" id="custom">&#9654;</span>
        <b slot="label-after" id="lab">NEW</b>
      </img-compare>`);
    const result = await page.evaluate(() => {
      const root = document.querySelector('#s').shadowRoot;
      const line = getComputedStyle(root.querySelector('.line'));
      const knob = getComputedStyle(root.querySelector('.knob'));
      const tag = getComputedStyle(root.querySelector('.tag-before'));
      return {
        lineColor: line.backgroundColor,
        lineWidth: line.width,
        knobBg: knob.backgroundColor,
        knobSize: knob.width,
        tagCase: tag.textTransform,
        slotted: root.querySelector('slot[name="handle"]').assignedElements()[0]?.id,
        labelSlot: root.querySelector('slot[name="label-after"]').assignedElements()[0]?.id,
      };
    });
    expect(result).toEqual({
      lineColor: 'rgb(255, 128, 0)',
      lineWidth: '6px',
      knobBg: 'rgb(0, 128, 0)',
      knobSize: '60px',
      tagCase: 'uppercase',
      slotted: 'custom',
      labelSlot: 'lab',
    });
    await page.close();
  });

  it('works inside another shadow root and survives being moved in the DOM', async () => {
    const page = await open('');
    const ok = await page.evaluate(
      async ({ red, blue }) => {
        const host = document.body.appendChild(document.createElement('div'));
        const shadow = host.attachShadow({ mode: 'open' });
        const el = shadow.appendChild(document.createElement('img-compare'));
        el.setAttribute('before-src', red);
        el.setAttribute('after-src', blue);
        el.position = 20;
        await new Promise((r) => (el.status === 'ready' ? r() : el.addEventListener('ready', r, { once: true })));
        document.body.append(host); // move
        return { status: el.status, position: el.position, ratio: el.shadowRoot.querySelector('.root').style.getPropertyValue('--ic-ratio') };
      },
      { red: RED, blue: BLUE },
    );
    expect(ok).toEqual({ status: 'ready', position: 20, ratio: '400 / 200' });
    await page.close();
  });
});

describe.skipIf(!chrome)('accessibility (axe-core)', () => {
  const axeSource = require.resolve('axe-core/axe.min.js');

  async function audit(page, selector = '#s') {
    await page.addScriptTag({ path: axeSource });
    return page.evaluate(async (sel) => {
      const { violations } = await window.axe.run(document.querySelector(sel));
      return violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(' ')).join(' | ')}`);
    }, selector);
  }

  it.each([
    ['default', ''],
    ['vertical', 'orientation="vertical"'],
    ['custom names', 'label="Colour grade" before-label="Gray" after-label="Colour"'],
    ['hover', 'hover'],
  ])('%s: no violations', async (_name, attrs) => {
    const page = await open(`<main>${SLIDER(attrs)}</main>`);
    expect(await audit(page)).toEqual([]);
    await page.close();
  });

  it('no violations while focused, in RTL, and in the error state', async () => {
    const page = await open(`<main>${SLIDER()}</main>`);
    await page.keyboard.press('Tab');
    expect(await audit(page)).toEqual([]);
    await page.evaluate(() => (document.documentElement.dir = 'rtl'));
    expect(await audit(page)).toEqual([]);
    await page.evaluate(() => (document.querySelector('#s').afterSrc = '/missing.png'));
    await page.waitForFunction(() => document.querySelector('#s').status === 'error');
    expect(await audit(page)).toEqual([]);
    await page.close();
  });

  it('exposes the live value to assistive tech', async () => {
    const page = await open(SLIDER('position="30" before-label="Gray" after-label="Colour"'));
    await page.keyboard.press('Tab');
    await page.keyboard.press('ArrowRight');
    const node = await page.evaluate(() => {
      const h = document.querySelector('#s').shadowRoot.querySelector('.handle');
      return ['role', 'aria-valuenow', 'aria-valuetext', 'aria-label', 'aria-orientation'].map((a) => h.getAttribute(a));
    });
    expect(node).toEqual(['slider', '31', '31% Gray, 69% Colour', 'Image comparison', 'horizontal']);
    await page.close();
  });
});

describe.skipIf(!chrome)('the published bundle', () => {
  it('loads in Node as CommonJS without a DOM (SSR), and exposes the same API', async () => {
    const cjs = require('../dist/index.cjs');
    expect(Object.keys(cjs).sort()).toEqual(
      ['ImgCompare', 'define', 'isSafeImageUrl', 'keyTarget', 'parsePosition', 'parseRatio', 'positionFromPoint', 'valueText'],
    );
    expect(cjs.define()).toBe(false);
  });

  it('the IIFE bundle works from a plain <script> tag', async () => {
    const page = await openWith(browser, server.url, '', { waitReady: false });
    await page.addScriptTag({ url: `${server.url}/dist/index.global.js` });
    expect(await page.evaluate(() => typeof window.ImgCompare?.ImgCompare)).toBe('function');
    await page.close();
  });
});
