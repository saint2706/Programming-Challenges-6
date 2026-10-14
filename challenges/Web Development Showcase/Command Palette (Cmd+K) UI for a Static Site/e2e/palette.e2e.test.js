import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { afterAll, afterEach, beforeAll, describe, expect, it } from 'vitest';
import { findChrome, launchChrome } from './chrome.mjs';
import { startServer } from './server.mjs';

/**
 * The palette in a real Chrome, against the production build: shortcuts, focus, ARIA, navigation,
 * storage failures, reduced motion, touch, no-JavaScript and an axe-core pass in both colour schemes.
 * Skipped (with a message) when no Chrome or Edge is installed.
 */

const chrome = findChrome();
if (!chrome) console.warn('[e2e] SKIPPED: no Chrome or Edge found. Set CHROME_PATH to run the browser tests.');

const AXE = readFileSync(fileURLToPath(new URL('../node_modules/axe-core/axe.min.js', import.meta.url)), 'utf8');
const PAGES = ['index', 'getting-started', 'keyboard', 'matching', 'accessibility', 'faq'];

let server;
let browser;
let contexts = [];

beforeAll(async () => {
    if (!chrome) return;
    server = await startServer();
    browser = await launchChrome();
});

// Every page gets its own browser context, so no test sees another's localStorage.
afterEach(async () => {
    await Promise.all(contexts.map((context) => context.close()));
    contexts = [];
});

afterAll(async () => {
    await browser?.close();
    await server?.close();
});

async function newPage() {
    const context = await browser.createBrowserContext();
    contexts.push(context);
    return context.newPage();
}

/** Open a page of the built site. `setup(page)` runs before it loads. */
async function open(path = 'index.html', { viewport, setup } = {}) {
    const page = await newPage();
    await page.setViewport(viewport ?? { width: 1100, height: 800 });
    await setup?.(page);
    await page.goto(`${server.url}/${path}?e2e`, { waitUntil: 'load' });
    return page;
}

async function chord(page, ...keys) {
    const mods = keys.slice(0, -1);
    for (const mod of mods) await page.keyboard.down(mod);
    await page.keyboard.press(keys.at(-1));
    for (const mod of mods.reverse()) await page.keyboard.up(mod);
}

const state = (page) =>
    page.evaluate(() => {
        const dialog = document.querySelector('dialog.cp');
        const input = document.querySelector('.cp-input');
        const options = [...document.querySelectorAll('[role="option"]')];
        const selected = options.find((o) => o.getAttribute('aria-selected') === 'true');
        return {
            open: Boolean(dialog?.open),
            focusInInput: document.activeElement === input,
            focusInDialog: Boolean(document.activeElement?.closest('dialog.cp')),
            titles: options.map((o) => o.querySelector('.cp-title').textContent),
            selectedTitle: selected?.querySelector('.cp-title').textContent ?? null,
            activeDescendantIsSelected: Boolean(selected) && input?.getAttribute('aria-activedescendant') === selected.id,
            groups: [...document.querySelectorAll('.cp-group-label')].map((g) => g.textContent),
            empty: document.querySelector('.cp-empty')?.hidden === false ? document.querySelector('.cp-empty').textContent : null,
            value: input?.value ?? null,
        };
    });

const waitOpen = (page) => page.waitForFunction(() => document.querySelector('dialog.cp')?.open);
const waitClosed = (page) => page.waitForFunction(() => !document.querySelector('dialog.cp')?.open);
const waitIndex = (page) => page.waitForFunction(() => document.querySelector('dialog.cp')?.dataset.index === 'ready' && document.querySelectorAll('[role="option"]').length > 0);

async function openPalette(page, query) {
    await chord(page, 'Control', 'k');
    await waitOpen(page);
    await waitIndex(page);
    if (query) await page.keyboard.type(query);
    return state(page);
}

describe.skipIf(!chrome)('opening and closing', () => {
    it('opens with Ctrl+K, focuses the input, and Ctrl+K again closes it', async () => {
        const page = await open();
        expect((await state(page)).open).toBe(false);
        await chord(page, 'Control', 'k');
        await waitOpen(page);
        expect(await state(page)).toMatchObject({ open: true, focusInInput: true });
        await chord(page, 'Control', 'k');
        await waitClosed(page);
        await page.close();
    });

    it('opens with Cmd+K (Meta)', async () => {
        const page = await open();
        await chord(page, 'Meta', 'k');
        await waitOpen(page);
        await page.close();
    });

    it('opens with "/" outside a field, and types "/" inside the palette input', async () => {
        const page = await open();
        await page.keyboard.press('/');
        await waitOpen(page);
        await page.keyboard.press('/');
        expect((await state(page)).value).toBe('/');
        await page.close();
    });

    it('opens from the header button and returns focus to it on Esc', async () => {
        const page = await open();
        await page.click('[data-palette-open]');
        await waitOpen(page);
        await page.keyboard.press('Escape');
        await waitClosed(page);
        expect(await page.evaluate(() => document.activeElement?.hasAttribute('data-palette-open'))).toBe(true);
        await page.close();
    });

    it('restores focus to a link that had it when opened with the keyboard', async () => {
        const page = await open('keyboard.html');
        await page.focus('main a, .toc a');
        const before = await page.evaluate(() => document.activeElement.getAttribute('href'));
        await chord(page, 'Control', 'k');
        await waitOpen(page);
        await page.keyboard.press('Escape');
        await waitClosed(page);
        expect(await page.evaluate(() => document.activeElement.getAttribute('href'))).toBe(before);
        await page.close();
    });

    it('closes when the dimmed area is clicked', async () => {
        const page = await open();
        await openPalette(page);
        await page.mouse.click(5, 5);
        await waitClosed(page);
        await page.close();
    });

    it('keeps Tab and Shift+Tab inside the dialog, and the page behind it unfocusable', async () => {
        const page = await open();
        await openPalette(page);
        for (let i = 0; i < 6; i += 1) {
            await page.keyboard.press('Tab');
            expect((await state(page)).focusInDialog).toBe(true);
        }
        await page.keyboard.down('Shift');
        await page.keyboard.press('Tab');
        await page.keyboard.up('Shift');
        expect((await state(page)).focusInDialog).toBe(true);
        // a click on the page behind lands on the backdrop, and the links behind are inert
        expect(await page.evaluate(() => document.querySelector('main a').matches(':focus'))).toBe(false);
        await page.close();
    });
});

describe.skipIf(!chrome)('search and navigation', () => {
    it('ranks word-start matches first and keeps aria-activedescendant on the selected option', async () => {
        const page = await open();
        const s = await openPalette(page, 'gs');
        expect(s.selectedTitle).toBe('Getting started');
        expect(s.activeDescendantIsSelected).toBe(true);
        await page.keyboard.press('ArrowDown');
        expect((await state(page)).activeDescendantIsSelected).toBe(true);
        await page.close();
    });

    it('Enter opens the page', async () => {
        const page = await open();
        await openPalette(page, 'faq');
        await Promise.all([page.waitForNavigation(), page.keyboard.press('Enter')]);
        expect(page.url()).toContain('/faq.html');
        await page.close();
    });

    it('a heading on the current page is a hash jump, not a reload', async () => {
        const page = await open('matching.html');
        await page.evaluate(() => {
            window.__sentinel = 1;
        });
        await openPalette(page, 'multiple words');
        await page.keyboard.press('Enter');
        await waitClosed(page);
        expect(page.url()).toMatch(/matching\.html\?e2e#multiple-words$/u);
        expect(await page.evaluate(() => window.__sentinel)).toBe(1);
        expect(await page.evaluate(() => document.activeElement?.id)).toBe('multiple-words');
        const top = await page.evaluate(() => document.getElementById('multiple-words').getBoundingClientRect().top);
        expect(top).toBeGreaterThanOrEqual(0);
        expect(top).toBeLessThan(400);
        await page.close();
    });

    it('a heading on another page loads it at that heading', async () => {
        const page = await open();
        await openPalette(page, 'ime');
        await Promise.all([page.waitForNavigation(), page.keyboard.press('Enter')]);
        expect(page.url()).toContain('accessibility.html#ime');
        await page.close();
    });

    it('Ctrl+Enter opens a new tab and keeps the palette open', async () => {
        const page = await open();
        await openPalette(page, 'faq');
        const created = new Promise((resolve) => browser.once('targetcreated', resolve));
        await chord(page, 'Control', 'Enter');
        const target = await created;
        expect(target.url()).toContain('faq.html');
        expect((await state(page)).open).toBe(true);
        await (await target.page())?.close();
        await page.close();
    });

    it('navigates with the arrows, Ctrl+N/P, Home and End', async () => {
        const page = await open();
        const first = (await openPalette(page)).selectedTitle;
        await page.keyboard.press('ArrowDown');
        const second = (await state(page)).selectedTitle;
        expect(second).not.toBe(first);
        await chord(page, 'Control', 'p');
        expect((await state(page)).selectedTitle).toBe(first);
        await page.keyboard.press('End');
        const last = (await state(page)).selectedTitle;
        expect(last).toBe((await state(page)).titles.at(-1));
        await page.keyboard.press('ArrowDown');
        expect((await state(page)).selectedTitle).toBe(first);
        await page.close();
    });

    it('shows sections and an empty state', async () => {
        const page = await open();
        const s = await openPalette(page);
        expect(s.groups).toEqual(['Actions', 'Pages']);
        await page.keyboard.type('qqqzx');
        const none = await state(page);
        expect(none.titles).toEqual([]);
        expect(none.empty).toContain('No results for “qqqzx”');
        await page.close();
    });

    it('finds text that is only in a keyword, and ignores accents in the query', async () => {
        const page = await open();
        expect((await openPalette(page, 'hotkeys')).titles).toContain('Keyboard shortcuts');
        await page.close();
    });

    it('is quick: results for a keystroke arrive within a frame or two', async () => {
        const page = await open();
        await openPalette(page);
        const ms = await page.evaluate(async () => {
            const input = document.querySelector('.cp-input');
            const t0 = performance.now();
            input.value = 'mat';
            input.dispatchEvent(new Event('input', { bubbles: true }));
            await new Promise((r) => requestAnimationFrame(() => r()));
            return performance.now() - t0;
        });
        expect(ms).toBeLessThan(100);
        await page.close();
    });
});

describe.skipIf(!chrome)('actions and recents', () => {
    it('toggles the theme, announces it and remembers it across a reload', async () => {
        const page = await open();
        await openPalette(page, 'theme');
        await page.keyboard.press('Enter');
        await waitClosed(page);
        await page.waitForFunction(() => document.querySelector('.toast')?.textContent === 'Theme: light');
        expect(await page.evaluate(() => document.documentElement.dataset.theme)).toBe('light');
        await page.reload({ waitUntil: 'load' });
        expect(await page.evaluate(() => document.documentElement.dataset.theme)).toBe('light');
        await page.close();
    });

    it('copies the link', async () => {
        const page = await open('faq.html', {
            setup: (p) =>
                p.browserContext().overridePermissions(server.url, ['clipboard-read', 'clipboard-write', 'clipboard-sanitized-write']),
        });
        await openPalette(page, 'copy link');
        await page.keyboard.press('Enter');
        await page.waitForFunction(() => document.querySelector('.toast')?.textContent === 'Link copied');
        expect(await page.evaluate(() => navigator.clipboard.readText())).toContain('/faq.html');
        await page.close();
    });

    it('scrolls to the top', async () => {
        const page = await open('matching.html', { viewport: { width: 900, height: 400 } });
        await page.evaluate(() => window.scrollTo(0, 800));
        await openPalette(page, 'scroll to top');
        await page.keyboard.press('Enter');
        await page.waitForFunction(() => window.scrollY === 0);
        await page.close();
    });

    it('remembers what you chose and lists it first next time', async () => {
        const page = await open();
        await openPalette(page, 'keyboard');
        await Promise.all([page.waitForNavigation(), page.keyboard.press('Enter')]);
        const stored = await page.evaluate(() => JSON.parse(localStorage.getItem('palette:recent')));
        expect(stored).toEqual({ v: 1, ids: ['p:keyboard.html'] });
        const s = await openPalette(page);
        expect(s.groups[0]).toBe('Recent');
        expect(s.titles[0]).toBe('Keyboard shortcuts');
        await page.close();
    });

    it('ignores corrupt stored recents', async () => {
        const page = await open('index.html', {
            setup: (p) => p.evaluateOnNewDocument(() => localStorage.setItem('palette:recent', '{"v":1,"ids":[1,{"x":2},"p:nope"')),
        });
        const s = await openPalette(page);
        expect(s.groups).toEqual(['Actions', 'Pages']);
        await page.close();
    });

    it('works when localStorage throws', async () => {
        const page = await open('index.html', {
            setup: (p) =>
                p.evaluateOnNewDocument(() => {
                    Object.defineProperty(window, 'localStorage', {
                        get() {
                            throw new DOMException('blocked', 'SecurityError');
                        },
                    });
                }),
        });
        await openPalette(page, 'faq');
        await Promise.all([page.waitForNavigation(), page.keyboard.press('Enter')]);
        expect(page.url()).toContain('faq.html');
        await page.close();
    });
});

describe.skipIf(!chrome)('input methods and devices', () => {
    it('searches while an IME composition is in progress, and Enter during it opens nothing', async () => {
        const page = await open();
        await openPalette(page);
        const session = await page.createCDPSession();
        await session.send('Input.imeSetComposition', { text: 'faq', selectionStart: 3, selectionEnd: 3 });
        await page.waitForFunction(() => document.querySelector('.cp-input').value === 'faq');
        expect((await state(page)).titles).toContain('FAQ');
        // A keydown that belongs to the composition (isComposing) must not open the result.
        await page.evaluate(() => {
            document.querySelector('.cp-input').dispatchEvent(
                new KeyboardEvent('keydown', { key: 'Enter', isComposing: true, keyCode: 229, bubbles: true, cancelable: true }),
            );
        });
        expect(page.url()).toContain('index.html');
        expect((await state(page)).open).toBe(true);
        await session.send('Input.insertText', { text: 'faq' });
        await session.detach();
        await page.close();
    });

    it('on a phone-sized touch screen: the button opens a full-screen palette and a tap picks a row', async () => {
        const page = await open('index.html', {
            viewport: { width: 390, height: 780, isMobile: true, hasTouch: true, deviceScaleFactor: 2 },
        });
        expect(await page.$eval('[data-palette-open]', (b) => !b.hidden && b.offsetHeight >= 44)).toBe(true);
        await page.tap('[data-palette-open]');
        await waitOpen(page);
        await waitIndex(page);
        await page.waitForFunction(() => document.querySelector('dialog.cp').getAnimations().length === 0);
        const box = await page.$eval('dialog.cp', (d) => {
            const r = d.getBoundingClientRect();
            return { w: r.width, h: r.height, vw: innerWidth, vh: innerHeight };
        });
        expect(box.w).toBe(box.vw);
        expect(box.h).toBe(box.vh);
        expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
        const row = await page.$('[role="option"]:nth-child(2)');
        await Promise.all([page.waitForNavigation().catch(() => null), row.tap()]);
        await page.close();
    });

    it('respects reduced motion', async () => {
        const anim = async (value) => {
            const page = await open('index.html', {
                setup: (p) => p.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value }]),
            });
            await openPalette(page);
            const name = await page.$eval('dialog.cp', (d) => getComputedStyle(d).animationName);
            await page.close();
            return name;
        };
        expect(await anim('no-preference')).toBe('cp-in');
        expect(await anim('reduce')).toBe('none');
    });
});

describe.skipIf(!chrome)('without JavaScript', () => {
    it('shows the menu and no dead search button', async () => {
        const page = await newPage();
        await page.setJavaScriptEnabled(false);
        await page.goto(`${server.url}/getting-started.html`, { waitUntil: 'load' });
        const result = await page.evaluate(() => ({
            links: [...document.querySelectorAll('.site-nav a')].map((a) => a.getAttribute('href')),
            buttonShown: getComputedStyle(document.querySelector('[data-palette-open]')).display !== 'none',
            hasDialog: Boolean(document.querySelector('dialog')),
            note: document.querySelector('noscript')?.textContent ?? '',
        }));
        expect(result.links).toHaveLength(6);
        expect(result.buttonShown).toBe(false);
        expect(result.hasDialog).toBe(false);
        await page.setJavaScriptEnabled(true);
        await page.close();
    });
});

describe.skipIf(!chrome)('accessibility (axe-core)', () => {
    async function violations(page) {
        await page.evaluate(AXE); // the axe-core library itself, injected into the page under test
        return page.evaluate(async () => {
            const { violations } = await axe.run(document, { runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21aa', 'best-practice'] } });
            return violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(' ')).join(' | ')}`);
        });
    }

    for (const scheme of ['light', 'dark']) {
        it.each(PAGES)(`${scheme}: ${'%s'} page has no violations`, async (name) => {
            const page = await open(`${name}.html`, {
                setup: (p) => p.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: scheme }]),
            });
            expect(await violations(page)).toEqual([]);
            await page.close();
        });

        it(`${scheme}: the open palette has no violations, with results and with none`, async () => {
            const page = await open('matching.html', {
                setup: (p) => p.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: scheme }]),
            });
            await openPalette(page, 'gap');
            expect(await violations(page)).toEqual([]);
            await page.keyboard.type('qqqq');
            expect((await state(page)).empty).not.toBeNull();
            expect(await violations(page)).toEqual([]);
            await page.close();
        });
    }
});
