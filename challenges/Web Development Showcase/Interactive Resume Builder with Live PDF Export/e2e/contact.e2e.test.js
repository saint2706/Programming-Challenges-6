import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { setPath } from '../src/lib/model.js';
import { MARGIN_MM } from '../src/lib/page.js';
import { findChrome, launchChrome } from './chrome.mjs';
import { longResume, openWith } from './fixtures.mjs';
import { readPdf, squash } from './pdf.mjs';
import { startServer } from './server.mjs';

/**
 * Email and phone in the printed PDF: the text is there, inside the margins, and each one is a live
 * link (`mailto:` / `tel:`) in the PDF itself, the way a recruiter's PDF reader will follow it.
 * Skipped (with a message) when no Chrome or Edge is installed.
 */

const chrome = findChrome();
if (!chrome) console.warn('[e2e] SKIPPED: no Chrome or Edge found. Set CHROME_PATH to run the print tests.');

const PT_PER_MM = 72 / 25.4;
const TEMPLATES = ['classic', 'sidebar', 'compact'];

let server;
let browser;

beforeAll(async () => {
  if (!chrome) return;
  server = await startServer();
  browser = await launchChrome();
});

afterAll(async () => {
  await browser?.close();
  await server?.close();
});

async function print(page) {
  await page.evaluate(() => window.dispatchEvent(new Event('beforeprint')));
  const shown = await page.evaluate(() => ({
    count: Number(document.querySelector('[data-page-count]').dataset.pageCount),
    texts: [...document.querySelectorAll('article.page')].map((article) => article.innerText),
  }));
  const bytes = await page.pdf({ preferCSSPageSize: true, printBackground: true });
  await page.evaluate(() => window.dispatchEvent(new Event('afterprint')));
  return { shown, pdf: await readPdf(bytes) };
}

const open = (doc, size) => openWith(browser, server.url, doc, size);
const linksOf = (pdf) => pdf.pages.flatMap((p) => p.links.map((l) => l.url));

describe.skipIf(!chrome)('email and phone in the PDF', () => {
  // The long example carries avery.morgan@example.com and +1 555 0100.
  for (const template of TEMPLATES) {
    it(`${template}: the address and the number print on the first page and are live links`, async () => {
      const page = await open(longResume(1, { template }));
      const { pdf } = await print(page);
      const first = squash(pdf.pages[0].text);
      expect(first).toContain('avery.morgan@example.com');
      expect(first).toContain('+15550100');
      expect(linksOf(pdf)).toContain('mailto:avery.morgan@example.com');
      expect(linksOf(pdf)).toContain('tel:+15550100');
      await page.close();
    });
  }

  it('puts only safe links in the PDF: no link of any other kind comes from the contact line', async () => {
    const page = await open(longResume(1, { template: 'classic' }));
    const { pdf } = await print(page);
    const urls = linksOf(pdf);
    expect(urls.length, 'the PDF has links to check').toBeGreaterThanOrEqual(2);
    for (const url of urls) expect(url, url).toMatch(/^(https?:|mailto:|tel:)/);
    await page.close();
  });

  it('keeps the longest valid address inside the margins and in the text, still one link', async () => {
    const address = `${'l'.repeat(64)}@${'d'.repeat(60)}.example.com`;
    let doc = longResume(1, { template: 'classic' });
    doc = setPath(doc, 'basics.email', address);
    const page = await open(doc);
    const { shown, pdf } = await print(page);
    expect(pdf.pages).toHaveLength(shown.count);
    expect(squash(pdf.pages[0].text)).toContain(address);
    expect(linksOf(pdf)).toContain(`mailto:${address}`);
    const side = MARGIN_MM.x * PT_PER_MM;
    for (const p of pdf.pages) {
      for (const item of p.items.filter((it) => it.str.trim())) {
        expect(item.x, item.str).toBeGreaterThanOrEqual(side - 1);
        expect(item.x + item.width, item.str).toBeLessThanOrEqual(p.width - side + 1);
      }
    }
    await page.close();
  });

  it('prints an address and a number typed in the side panel', async () => {
    const page = await open(longResume(1, { template: 'sidebar' }));
    await page.evaluate(() => {
      const set = (name, value) => {
        const label = [...document.querySelectorAll('label')].find((l) => l.textContent.trim() === name);
        const input = document.getElementById(label.htmlFor);
        input.value = value;
        input.dispatchEvent(new Event('input', { bubbles: true }));
      };
      set('Email', 'zed.zebra@example.org');
      set('Phone', '+44 20 7946 0958');
    });
    const { pdf } = await print(page);
    const first = squash(pdf.pages[0].text);
    expect(first).toContain('zed.zebra@example.org');
    expect(first).toContain('+442079460958');
    expect(linksOf(pdf)).toContain('mailto:zed.zebra@example.org');
    expect(linksOf(pdf)).toContain('tel:+442079460958');
    await page.close();
  });
});
