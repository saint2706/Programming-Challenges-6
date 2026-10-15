import { readFileSync } from 'node:fs';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { launchChrome } from './chrome.mjs';
import { startServer } from './server.mjs';

// Real-browser checks against the built site (dist/). Run with `pnpm run test:e2e`.
const AXE = readFileSync(new URL('../node_modules/axe-core/axe.min.js', import.meta.url), 'utf8');
const KEY = 'scroll-story-engine';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let browser;
let server;

beforeAll(async () => {
    browser = await launchChrome();
    server = await startServer();
}, 60_000);

afterAll(async () => {
    await browser?.close();
    await server?.close();
});

/**
 * Open the page in a fresh incognito-like context.
 * @param {{ engine?: 'css' | 'gsap', reduce?: boolean, dark?: boolean, width?: number, height?: number, js?: boolean }} opts
 */
async function open({ engine = 'css', reduce = false, dark = false, width = 1280, height = 800, js = true } = {}) {
    const context = await browser.createBrowserContext();
    const page = await context.newPage();
    await page.setViewport({ width, height });
    await page.setJavaScriptEnabled(js);
    await page.emulateMediaFeatures([
        { name: 'prefers-reduced-motion', value: reduce ? 'reduce' : 'no-preference' },
        { name: 'prefers-color-scheme', value: dark ? 'dark' : 'light' },
    ]);
    if (js) {
        await page.evaluateOnNewDocument((key, value) => {
            try {
                localStorage.setItem(key, value);
            } catch {
                /* storage blocked: the page must still work */
            }
        }, KEY, engine);
    }
    await page.goto(server.url, { waitUntil: 'load' });
    if (js) {
        await page.waitForFunction(() => document.documentElement.dataset.engine, { timeout: 15_000 });
        // The GSAP chunk loads after the engine flag is set. Wait for the chunk, then (with motion) for its triggers.
        if (engine === 'gsap') {
            await page.waitForFunction(() => performance.getEntriesByType('resource').some((e) => e.name.includes('gsap-engine')), { timeout: 15_000 });
            if (!reduce) await page.waitForFunction(() => globalThis.__storyDebug?.triggerCount() > 0, { timeout: 15_000 });
        }
    }
    return { context, page };
}

/** Scroll to y and wait until the scroll-linked output has settled (GSAP scrub lags by about 1 s). */
async function scrollTo(page, y, { settle = 1400 } = {}) {
    await page.evaluate((top) => window.scrollTo({ top, behavior: 'instant' }), y);
    await sleep(settle);
}

const absTop = (page, selector) =>
    page.evaluate((s) => {
        const el = document.querySelector(s);
        return el.getBoundingClientRect().top + window.scrollY;
    }, selector);

const stageTop = (page) =>
    page.evaluate(() => document.querySelector('#chart-stage').getBoundingClientRect().top);

const dashOffset = (page) =>
    page.evaluate(() => parseFloat(getComputedStyle(document.querySelector('.chart-line')).strokeDashoffset));

const currentStep = (page) =>
    page.evaluate(() => document.querySelector('[data-step][aria-current]')?.id ?? null);

describe.each(['css', 'gsap'])('pinned story (%s engine)', (engine) => {
    it('the chart stage stays pinned at the top through its scroll range', async () => {
        const { context, page } = await open({ engine });
        try {
            const start = await absTop(page, '#chart-story');
            const end = start + (await page.evaluate(() => document.querySelector('#chart-story').offsetHeight));
            const vh = await page.evaluate(() => window.innerHeight);
            const samples = [0.1, 0.3, 0.5, 0.7, 0.9];
            for (const f of samples) {
                await scrollTo(page, start + (end - start - vh) * f, { settle: 400 });
                const top = await stageTop(page);
                expect(Math.abs(top), `stage top at ${f}`).toBeLessThanOrEqual(2);
            }
        } finally {
            await context.close();
        }
    }, 60_000);

    it('the path draws as the reader scrolls: strokeDashoffset falls monotonically to zero', async () => {
        const { context, page } = await open({ engine });
        try {
            const start = await absTop(page, '#chart-story');
            const end = start + (await page.evaluate(() => document.querySelector('#chart-story').offsetHeight));
            const vh = await page.evaluate(() => window.innerHeight);
            const values = [];
            for (const f of [0, 0.25, 0.5, 0.75, 1]) {
                await scrollTo(page, start + (end - start - vh) * f);
                values.push(await dashOffset(page));
            }
            for (let i = 1; i < values.length; i++) expect(values[i]).toBeLessThanOrEqual(values[i - 1] + 1e-6);
            expect(values[0]).toBeGreaterThan(values.at(-1));
            expect(values.at(-1)).toBeLessThan(1);
        } finally {
            await context.close();
        }
    }, 60_000);

    it('the active step moves forward in order as the reader scrolls', async () => {
        const { context, page } = await open({ engine });
        try {
            const start = await absTop(page, '#chart-story');
            const end = start + (await page.evaluate(() => document.querySelector('#chart-story').offsetHeight));
            const vh = await page.evaluate(() => window.innerHeight);
            const seen = [];
            for (let i = 0; i <= 20; i++) {
                await scrollTo(page, start + ((end - start - vh) * i) / 20, { settle: 250 });
                const id = await currentStep(page);
                if (id && seen.at(-1) !== id) seen.push(id);
            }
            expect(seen).toEqual(['step-1', 'step-2', 'step-3', 'step-4', 'step-5']);
        } finally {
            await context.close();
        }
    }, 90_000);
});

describe('engine toggle', () => {
    it('switches engines in place without a reload and never duplicates triggers', async () => {
        const { context, page } = await open({ engine: 'css' });
        try {
            await page.evaluate(() => {
                globalThis.__marker = 'same-document';
            });
            expect(await page.evaluate(() => globalThis.__storyDebug.triggerCount())).toBe(0);

            await page.click('#engine-toggle');
            await page.waitForFunction(() => globalThis.__storyDebug.triggerCount() > 0, { timeout: 15_000 });
            const first = await page.evaluate(() => globalThis.__storyDebug.triggerCount());
            expect(first).toBeGreaterThan(0);
            expect(await page.evaluate(() => document.documentElement.dataset.engine)).toBe('gsap');

            for (let i = 0; i < 2; i++) {
                await page.click('#engine-toggle');
                await page.waitForFunction(() => globalThis.__storyDebug.engine === 'css');
                expect(await page.evaluate(() => globalThis.__storyDebug.triggerCount())).toBe(0);
                await page.click('#engine-toggle');
                await page.waitForFunction((n) => globalThis.__storyDebug.triggerCount() === n, { timeout: 15_000 }, first);
                expect(await page.evaluate(() => globalThis.__storyDebug.triggerCount())).toBe(first);
            }

            expect(await page.evaluate(() => globalThis.__marker)).toBe('same-document');
            expect(await page.evaluate(() => localStorage.getItem('scroll-story-engine'))).toBe('gsap');
        } finally {
            await context.close();
        }
    }, 90_000);
});

describe('reduced motion', () => {
    it('loads no scroll engine: no triggers, a static layout, and the path stays fully drawn', async () => {
        const { context, page } = await open({ engine: 'gsap', reduce: true });
        try {
            expect(await page.evaluate(() => globalThis.__storyDebug.triggerCount())).toBe(0);
            const position = await page.evaluate(() => getComputedStyle(document.querySelector('#chart-stage')).position);
            expect(position).not.toBe('sticky');
            const before = await stageTop(page);
            await page.evaluate(() => window.scrollBy(0, 600));
            await sleep(300);
            expect(await stageTop(page)).toBeLessThan(before);
            expect(await dashOffset(page)).toBeCloseTo(0, 3);
        } finally {
            await context.close();
        }
    }, 60_000);
});

describe('without JavaScript', () => {
    it('the text, the chart line, the stripes and the data table are all present', async () => {
        const { context, page } = await open({ js: false });
        try {
            const facts = await page.evaluate(() => ({
                heading: document.querySelector('h1').textContent.trim(),
                pathLength: document.querySelector('.chart-line').getTotalLength(),
                stripes: document.querySelectorAll('.stripe').length,
                tableRows: document.querySelectorAll('.data-table tbody tr').length,
                stepsVisible: [...document.querySelectorAll('.step')].every((s) => s.getBoundingClientRect().height > 0),
                toggleHidden: document.querySelector('#engine-toggle').hidden,
            }));
            expect(facts.heading).toContain('The world has warmed');
            expect(facts.pathLength).toBeGreaterThan(1000);
            expect(facts.stripes).toBe(146);
            expect(facts.tableRows).toBe(146);
            expect(facts.stepsVisible).toBe(true);
            expect(facts.toggleHidden).toBe(true);
        } finally {
            await context.close();
        }
    }, 60_000);
});

describe('narrow phones', () => {
    it('at 360px wide the page does not scroll sideways', async () => {
        for (const engine of ['css', 'gsap']) {
            const { context, page } = await open({ engine, width: 360, height: 740 });
            try {
                const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
                expect(overflow, engine).toBeLessThanOrEqual(0);
            } finally {
                await context.close();
            }
        }
    }, 90_000);
});

describe('keyboard', () => {
    it('the first Tab reaches a skip link that jumps to the end of the page', async () => {
        const { context, page } = await open();
        try {
            await page.keyboard.press('Tab');
            const focused = await page.evaluate(() => ({
                cls: document.activeElement.className,
                text: document.activeElement.textContent.trim(),
            }));
            expect(focused.cls).toBe('skip');
            expect(focused.text).toBe('Skip to the end');
            await page.keyboard.press('Enter');
            await sleep(200);
            expect(await page.evaluate(() => location.hash)).toBe('#end');
        } finally {
            await context.close();
        }
    }, 60_000);
});

describe('axe-core accessibility', () => {
    const scan = async (page) => {
        await page.evaluate(AXE);
        return page.evaluate(async () => {
            const result = await axe.run(document, { runOnly: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'] });
            return result.violations.map((v) => ({ id: v.id, impact: v.impact, nodes: v.nodes.length }));
        });
    };

    for (const engine of ['css', 'gsap']) {
        for (const dark of [false, true]) {
            it(`has no WCAG 2.1 AA violations (${engine}, ${dark ? 'dark' : 'light'})`, async () => {
                const { context, page } = await open({ engine, dark });
                try {
                    expect(await scan(page)).toEqual([]);
                } finally {
                    await context.close();
                }
            }, 60_000);
        }
    }
});
