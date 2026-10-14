// Takes the README screenshots into docs/. Run `npm run screenshots` (builds first, needs Chrome or Edge).
import { launchChrome } from './chrome.mjs';
import { openApp, typeValue } from './helpers.mjs';
import { startServer } from './server.mjs';

const server = await startServer();
const browser = await launchChrome();
const preset = (page, name) => page.evaluate((n) => [...document.querySelectorAll('#presets button')].find((b) => b.textContent === n).click(), name);
const settle = (ms = 700) => new Promise((r) => setTimeout(r, ms));

let { page } = await openApp(browser, server.url, { light: true, width: 1400, height: 1000, reduceMotion: true });
await page.screenshot({ path: 'docs/default.png' });
await page.close();

for (const name of ['Forest', 'Sunset', 'Mono']) {
  ({ page } = await openApp(browser, server.url, { light: true, width: 1400, height: 1000, reduceMotion: true }));
  await preset(page, name);
  await settle(100);
  await (await page.$('#panes')).screenshot({ path: `docs/${name.toLowerCase()}.png` });
  await page.close();
}

({ page } = await openApp(browser, server.url, { light: true, width: 1400, height: 1000, reduceMotion: true }));
await typeValue(page, 'brandL', 0.75);
await (await page.$('#contrast-region')).screenshot({ path: 'docs/contrast-fail.png' });
await page.click('tr[data-pair="brand-surface"][data-scheme="light"] .fix');
await settle(100);
await (await page.$('#contrast-region')).screenshot({ path: 'docs/contrast-fixed.png' });
await page.close();

({ page } = await openApp(browser, server.url, { light: true, width: 390, height: 844, reduceMotion: true }));
await page.screenshot({ path: 'docs/mobile.png' });
await page.close();

await browser.close();
await server.close();
