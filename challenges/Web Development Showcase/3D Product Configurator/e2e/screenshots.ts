// `pnpm run screenshots`: rewrites docs/*.png from the production build.
import { mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { launchChrome } from './chrome.ts';
import { openStudio, read, startServer } from './support.ts';

const docs = fileURLToPath(new URL('../docs/', import.meta.url));
mkdirSync(docs, { recursive: true });
const server = await startServer();
const browser = await launchChrome();
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function shoot(name: string, options: { query?: string; width?: number; height?: number; scheme?: 'light' | 'dark'; eye?: { x: number; y: number; z: number }; before?: (page: import('puppeteer-core').Page) => Promise<void> }) {
    const { page } = await openStudio(browser, server.url, { query: options.query, width: options.width ?? 1280, height: options.height ?? 800 });
    await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: options.scheme ?? 'dark' }]);
    if (options.eye) await page.evaluate((eye) => ((window as unknown as { __app: { studio: { eyeOverride: unknown } } }).__app.studio.eyeOverride = eye), options.eye);
    await options.before?.(page);
    await sleep(900);
    await page.screenshot({ path: `${docs}${name}.png` });
    console.log(name, await read(page, 'studio.backend'));
    await page.close();
}

await shoot('design-dark', {});
await shoot('design-light', { scheme: 'light', before: (page) => page.evaluate(() => (window as unknown as { __app: { studio: { applyPreset(id: string): void } } }).__app.studio.applyPreset('ember')) });
await shoot('window-left', { query: '&mode=window', eye: { x: -0.1, y: 0.02, z: 0.55 } });
await shoot('window-right', { query: '&mode=window', eye: { x: 0.1, y: 0.02, z: 0.55 } });
await shoot('design-phone', { width: 390, height: 844 });

await browser.close();
await server.close();
