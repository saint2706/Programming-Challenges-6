// Benchmarks in a real Chrome: how long a repagination takes (cold and with the height cache), and
// how long from a keystroke to the repainted page. Run with `npm run bench`.
import { findChrome, launchChrome } from './chrome.mjs';
import { longResume, openWith } from './fixtures.mjs';
import { startServer } from './server.mjs';
import { seedResume } from '../src/lib/seed.js';

if (!findChrome()) {
  console.warn('[bench] SKIPPED: no Chrome or Edge found. Set CHROME_PATH to run the benchmark.');
  process.exit(0);
}

const RUNS = 30;
const KEYSTROKES = 120;

const percentile = (values, p) => {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1)];
};
const ms = (n) => `${n.toFixed(1)} ms`;

const FIXTURES = [
  { label: 'my resume (sidebar)', doc: seedResume() },
  { label: 'long example (compact)', doc: longResume(1, { template: 'compact' }) },
  { label: 'long example x5 (classic)', doc: longResume(5, { template: 'classic' }) },
  { label: 'long example x16 (classic)', doc: longResume(16, { template: 'classic' }) },
];

const server = await startServer();
const browser = await launchChrome();
try {
  console.log(`Chrome: ${await browser.version()}   (${RUNS} runs each, ${KEYSTROKES} keystrokes)\n`);
  console.log('Repagination (measure + paginate + render)');
  console.log('fixture                      pages  blocks   cold p50     cold p95     cached p50   cached p95');

  for (const { label, doc } of FIXTURES) {
    const page = await openWith(browser, server.url, doc);
    const result = await page.evaluate(async (runs) => {
      const cold = [];
      const cached = [];
      for (let i = 0; i < runs; i++) cold.push(window.__resume.repaginate({ cold: true }));
      for (let i = 0; i < runs; i++) cached.push(window.__resume.repaginate());
      return {
        cold,
        cached,
        pages: Number(document.querySelector('[data-page-count]').dataset.pageCount),
        blocks: document.querySelectorAll('article [data-block]').length,
      };
    }, RUNS);
    console.log(
      `${label.padEnd(28)} ${String(result.pages).padStart(5)}  ${String(result.blocks).padStart(6)}   ` +
        `${ms(percentile(result.cold, 50)).padEnd(12)} ${ms(percentile(result.cold, 95)).padEnd(12)} ` +
        `${ms(percentile(result.cached, 50)).padEnd(12)} ${ms(percentile(result.cached, 95))}`,
    );
    await page.close();
  }

  console.log('\nKeystroke to repainted page (typing in a bullet on page 1)');
  console.log('fixture                      pages   p50          p95          max          blocks re-measured per key');
  for (const { label, doc } of FIXTURES.slice(1)) {
    const page = await openWith(browser, server.url, doc);
    await page.evaluate(() => {
      const el = document.querySelector('article.page [contenteditable][data-path^="work.0.highlights"]');
      el.focus();
      const range = document.createRange();
      range.selectNodeContents(el);
      range.collapse(false);
      getSelection().removeAllRanges();
      getSelection().addRange(range);
      window.__latency = [];
      window.__measuredPerKey = [];
      document.addEventListener(
        'input',
        () => {
          const started = performance.now();
          requestAnimationFrame(() =>
            requestAnimationFrame(() => {
              window.__latency.push(performance.now() - started);
              window.__measuredPerKey.push(window.__resume.stats().measured);
            }),
          );
        },
        true,
      );
    });
    for (let i = 0; i < KEYSTROKES; i++) {
      await page.keyboard.type(i % 12 === 11 ? ' ' : 'x');
      await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
    }
    const data = await page.evaluate(() => ({
      latency: window.__latency,
      measured: window.__measuredPerKey,
      pages: Number(document.querySelector('[data-page-count]').dataset.pageCount),
    }));
    const avgMeasured = data.measured.reduce((a, b) => a + b, 0) / data.measured.length;
    console.log(
      `${label.padEnd(28)} ${String(data.pages).padStart(5)}   ${ms(percentile(data.latency, 50)).padEnd(12)} ` +
        `${ms(percentile(data.latency, 95)).padEnd(12)} ${ms(Math.max(...data.latency)).padEnd(12)} ${avgMeasured.toFixed(2)}`,
    );
    await page.close();
  }
} finally {
  await browser.close();
  await server.close();
}
