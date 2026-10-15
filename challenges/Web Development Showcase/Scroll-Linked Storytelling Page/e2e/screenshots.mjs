// Regenerates the screenshots in docs/ from the built site. Run `pnpm run build` first, or use
// `pnpm run screenshots`, which builds and then runs this file.
import { fileURLToPath } from 'node:url';
import { findChrome, launchChrome } from './chrome.mjs';
import { startServer } from './server.mjs';

if (!findChrome()) {
    console.warn('[screenshots] SKIPPED: no Chrome or Edge found. Set CHROME_PATH to run this.');
    process.exit(0);
}

const out = (name) => fileURLToPath(new URL(`../docs/${name}`, import.meta.url));
const server = await startServer();
const browser = await launchChrome();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Scroll to `fraction` of the chart's pinned range, so the chart shows a mid-story state. */
async function scrollToChart(page, fraction) {
    await page.evaluate((f) => {
        const story = document.querySelector('#chart-story');
        const top = story.getBoundingClientRect().top + window.scrollY;
        const span = story.offsetHeight - window.innerHeight;
        window.scrollTo({ top: top + span * f, behavior: 'instant' });
    }, fraction);
    await sleep(700);
}

async function shot({ name, scheme = 'light', viewport, fraction = null }) {
    const context = await browser.createBrowserContext();
    const page = await context.newPage();
    await page.setViewport({ ...viewport, deviceScaleFactor: viewport.isMobile ? 2 : 1 });
    await page.emulateMediaFeatures([
        { name: 'prefers-color-scheme', value: scheme },
        { name: 'prefers-reduced-motion', value: 'no-preference' },
    ]);
    await page.goto(server.url, { waitUntil: 'load' });
    await page.waitForFunction(() => document.documentElement.dataset.engine === 'css');
    if (fraction !== null) await scrollToChart(page, fraction);
    await page.screenshot({ path: out(name) });
    await context.close();
    console.log('wrote', name);
}

try {
    await shot({ name: 'story-light.png', viewport: { width: 1000, height: 640 }, fraction: 0.55 });
    await shot({ name: 'story-dark.png', scheme: 'dark', viewport: { width: 1000, height: 640 }, fraction: 0.85 });
    await shot({ name: 'story-mobile-360.png', viewport: { width: 360, height: 740, isMobile: true, hasTouch: true }, fraction: 0.5 });
    await shot({ name: 'story-mobile-390.png', viewport: { width: 390, height: 780, isMobile: true, hasTouch: true } });
} finally {
    await browser.close();
    await server.close();
}
