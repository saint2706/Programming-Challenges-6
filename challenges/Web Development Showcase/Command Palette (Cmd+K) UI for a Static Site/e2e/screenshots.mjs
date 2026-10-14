// Regenerates the screenshots in docs/ from the built site. Run `npm run build` first, then
// `node e2e/screenshots.mjs`.
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

async function shot({ name, path = 'index.html', query = '', scheme = 'light', viewport = { width: 1000, height: 560 }, keys = [] }) {
    const context = await browser.createBrowserContext();
    const page = await context.newPage();
    await page.setViewport({ ...viewport, deviceScaleFactor: viewport.isMobile ? 2 : 1 });
    await page.emulateMediaFeatures([
        { name: 'prefers-color-scheme', value: scheme },
        { name: 'prefers-reduced-motion', value: 'reduce' },
    ]);
    await page.goto(`${server.url}/${path}`, { waitUntil: 'load' });
    await page.keyboard.down('Control');
    await page.keyboard.press('k');
    await page.keyboard.up('Control');
    await page.waitForFunction(() => document.querySelector('dialog.cp')?.dataset.index === 'ready');
    if (query) await page.keyboard.type(query);
    for (const key of keys) await page.keyboard.press(key);
    await new Promise((resolve) => setTimeout(resolve, 300));
    await page.screenshot({ path: out(name) });
    await context.close();
    console.log('wrote', name);
}

try {
    await shot({ name: 'palette-light.png', query: 'gs', path: 'matching.html' });
    await shot({ name: 'palette-dark.png', query: 'mw', scheme: 'dark', path: 'keyboard.html', keys: ['ArrowDown'] });
    await shot({ name: 'palette-empty-query.png', path: 'faq.html' });
    await shot({ name: 'palette-no-results.png', query: 'qqqzx', scheme: 'dark', viewport: { width: 1000, height: 360 } });
    await shot({ name: 'palette-mobile.png', query: 'acc', viewport: { width: 390, height: 780, isMobile: true, hasTouch: true } });
} finally {
    await browser.close();
    await server.close();
}
