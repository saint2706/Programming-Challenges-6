// Honest numbers for the README: bundle size, per-pointermove cost, frame pacing during a real
// mouse drag (idle vs drag, and with the CPU throttled), and the cost of creating many sliders.
// Run `npm run bench` (builds first). Numbers are from headless Chrome on whatever machine runs it.
import { readFileSync } from 'node:fs';
import { brotliCompressSync, gzipSync } from 'node:zlib';
import { launchChrome } from './chrome.mjs';
import { RED, BLUE, SLIDER, boxOf, openWith } from './fixtures.mjs';
import { startServer } from './server.mjs';

const median = (a) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];
const pct = (a, p) => [...a].sort((x, y) => x - y)[Math.min(a.length - 1, Math.floor((a.length * p) / 100))];
const fmt = (n, d = 2) => Number(n.toFixed(d));

const bundle = readFileSync(new URL('../dist/index.js', import.meta.url));
console.log('bundle (dist/index.js, minified ESM)', {
  bytes: bundle.length,
  gzip: gzipSync(bundle, { level: 9 }).length,
  brotli: brotliCompressSync(bundle).length,
});

const server = await startServer();
const browser = await launchChrome();
console.log('browser', await browser.version());

// 1. Cost of one pointermove: synthetic events on the shadow root, so it is the handler plus the
// style/layout work its write forces (getBoundingClientRect on the next event), without paint.
{
  const page = await openWith(browser, server.url, SLIDER());
  const runs = [];
  for (let run = 0; run < 7; run++) {
    runs.push(
      await page.evaluate(() => {
        const root = document.querySelector('#s').shadowRoot.querySelector('.root');
        const box = root.getBoundingClientRect();
        const fire = (type, x) =>
          root.dispatchEvent(new PointerEvent(type, { pointerId: 1, pointerType: 'mouse', clientX: x, clientY: box.top + 10, bubbles: true, button: 0, buttons: 1 }));
        fire('pointerdown', box.left + 5);
        const times = [];
        const start = performance.now();
        for (let i = 0; i < 2000; i++) {
          const x = box.left + 5 + ((i * 7) % (box.width - 10));
          const t = performance.now();
          fire('pointermove', x);
          times.push(performance.now() - t);
        }
        const mean = (performance.now() - start) / 2000;
        fire('pointerup', box.left + 5);
        return { times, mean };
      }),
    );
  }
  const all = runs.flatMap((r) => r.times);
  console.log('pointermove handler (ms, 7 x 2000 events; timer resolution is 0.1 ms)', {
    meanFromWholeLoop: fmt(median(runs.map((r) => r.mean)), 4),
    median: fmt(median(all), 4),
    p95: fmt(pct(all, 95), 4),
    p99: fmt(pct(all, 99), 4),
    max: fmt(Math.max(...all), 3),
  });
  await page.close();
}

// 2. Frame pacing during a real CDP mouse drag.
async function framePacing(throttle) {
  const page = await openWith(browser, server.url, `<div style="height:40px"></div>${SLIDER()}`);
  const client = await page.createCDPSession();
  if (throttle) await client.send('Emulation.setCPUThrottlingRate', { rate: throttle });
  await page.evaluate(() => {
    window.__frames = [];
    let last = performance.now();
    const tick = (now) => {
      window.__frames.push(now - last);
      last = now;
      if (!window.__stop) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
  const measure = async (drag) => {
    await page.evaluate(() => (window.__frames.length = 0));
    if (drag) {
      const box = await boxOf(page);
      const y = box.y + box.height * 0.8;
      await page.mouse.move(box.x + 10, y);
      await page.mouse.down();
      for (let pass = 0; pass < 4; pass++) {
        for (let i = 0; i <= 60; i++) {
          const f = pass % 2 ? 1 - i / 60 : i / 60;
          await page.mouse.move(box.x + 10 + f * (box.width - 20), y);
        }
      }
      await page.mouse.up();
    } else {
      await new Promise((r) => setTimeout(r, 2000));
    }
    const frames = (await page.evaluate(() => window.__frames.slice())).slice(2);
    return { frames: frames.length, medianMs: fmt(median(frames)), p95Ms: fmt(pct(frames, 95)), maxMs: fmt(Math.max(...frames)), over25ms: frames.filter((f) => f > 25).length };
  };
  const idle = await measure(false);
  const drag = await measure(true);
  await page.evaluate(() => (window.__stop = true));
  await page.close();
  return { idle, drag };
}
console.log('frame pacing, no throttle', await framePacing(0));
console.log('frame pacing, 4x CPU throttle', await framePacing(4));

// 3. Creating many sliders (parse, upgrade, attach shadow root, adopt the shared stylesheet).
{
  const page = await openWith(browser, server.url, '');
  const ms = await page.evaluate(
    ({ red, blue }) => {
      const t = performance.now();
      const frag = document.createDocumentFragment();
      for (let i = 0; i < 200; i++) {
        const el = document.createElement('img-compare');
        el.setAttribute('before-src', red);
        el.setAttribute('after-src', blue);
        el.setAttribute('loading', 'eager');
        frag.append(el);
      }
      document.body.append(frag);
      void document.body.offsetHeight; // force layout
      return performance.now() - t;
    },
    { red: RED, blue: BLUE },
  );
  console.log('create + attach + lay out 200 sliders (ms)', fmt(ms, 1), `(${fmt(ms / 200, 3)} ms each)`);
  await page.close();
}

await browser.close();
await server.close();
