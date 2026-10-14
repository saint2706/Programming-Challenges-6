// Regenerates docs/*.webp from the demo page. Run `npm run screenshots` (builds first).
import { mkdir } from 'node:fs/promises';
import { launchChrome } from './chrome.mjs';
import { startServer } from './server.mjs';

await mkdir('docs', { recursive: true });
const server = await startServer();
const browser = await launchChrome();
const shot = (page, name, options = {}) => page.screenshot({ path: `docs/${name}.webp`, type: 'webp', quality: 82, ...options });
const settle = () => new Promise((r) => setTimeout(r, 400));

async function demo(width, height) {
  const page = await browser.newPage();
  await page.setViewport({ width, height, deviceScaleFactor: 1 });
  await page.goto(`${server.url}/demo/index.html`);
  await page.waitForFunction(() => [...document.querySelectorAll('img-compare')].every((el) => el.status === 'ready' || el.status === 'loading') && customElements.get('img-compare'));
  // The demo is lazy: scroll through so every image is fetched, then back.
  await page.evaluate(async () => {
    for (let y = 0; y < document.body.scrollHeight; y += 400) {
      scrollTo(0, y);
      await new Promise((r) => setTimeout(r, 120));
    }
    scrollTo(0, 0);
  });
  await page.waitForFunction(() => [...document.querySelectorAll('img-compare')].every((el) => el.status === 'ready'));
  await settle();
  return page;
}

let page = await demo(1000, 900);
await shot(page, 'demo-desktop', { fullPage: true });
const slider = await page.$('#main');
await page.focus('#main'); // focus lands on the host's handle via Tab order below instead
await page.keyboard.press('Tab');
await page.keyboard.press('ArrowRight');
await settle();
await shot(page, 'focus-ring', { clip: { ...(await slider.boundingBox()) } });
await page.evaluate(() => {
  document.documentElement.dir = 'rtl';
});
await settle();
await shot(page, 'rtl', { clip: { ...(await slider.boundingBox()) } });
await page.close();

page = await demo(390, 800);
await shot(page, 'demo-mobile', { fullPage: true });
await page.close();

await browser.close();
await server.close();
console.log('docs written');
