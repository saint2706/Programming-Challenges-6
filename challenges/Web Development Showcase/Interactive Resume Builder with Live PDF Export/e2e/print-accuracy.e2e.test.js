import { createRequire } from 'node:module';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { setPath } from '../src/lib/model.js';
import { MARGIN_MM } from '../src/lib/page.js';
import { findChrome, launchChrome } from './chrome.mjs';
import { longResume, openWith } from './fixtures.mjs';
import { readPdf, squash } from './pdf.mjs';
import { startServer } from './server.mjs';

/**
 * Does the printed PDF match the preview? These run against the built app in a real Chrome: they
 * make a PDF the way "Save as PDF" does, read it back with pdf.js, and compare it with what the
 * page showed. Skipped (with a message) when no Chrome or Edge is installed.
 */

const chrome = findChrome();
if (!chrome) console.warn('[e2e] SKIPPED: no Chrome or Edge found. Set CHROME_PATH to run the print tests.');

const PT_PER_MM = 72 / 25.4;
const SIZES = { a4: [595.28, 841.89], letter: [612, 792] };
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

/** Make the preview current, snapshot what it shows, then print it to a PDF. */
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

describe.skipIf(!chrome)('the PDF matches the preview', () => {
  for (const template of TEMPLATES) {
    for (const pageSize of ['a4', 'letter']) {
      it(`${template} on ${pageSize}: same pages, same size, same text per page, inside the margins`, async () => {
        const page = await open(longResume(1, { template, pageSize }));
        const { shown, pdf } = await print(page);

        expect(shown.count).toBeGreaterThanOrEqual(3);
        expect(pdf.pages).toHaveLength(shown.count);

        const [width, height] = SIZES[pageSize];
        for (const p of pdf.pages) {
          expect(Math.abs(p.width - width)).toBeLessThan(1.5);
          expect(Math.abs(p.height - height)).toBeLessThan(1.5);
        }

        pdf.pages.forEach((p, i) => {
          expect(squash(p.text), `text of page ${i + 1}`).toBe(squash(shown.texts[i]));
        });

        const side = MARGIN_MM.x * PT_PER_MM;
        const top = MARGIN_MM.top * PT_PER_MM;
        const bottom = MARGIN_MM.bottom * PT_PER_MM;
        for (const [i, p] of pdf.pages.entries()) {
          for (const item of p.items.filter((it) => it.str.trim())) {
            const where = `page ${i + 1} "${item.str.slice(0, 30)}"`;
            expect(item.x, `${where} left`).toBeGreaterThanOrEqual(side - 1);
            expect(item.x + item.width, `${where} right`).toBeLessThanOrEqual(p.width - side + 1);
            expect(item.y + item.height * 0.75, `${where} top`).toBeLessThanOrEqual(p.height - top + 2);
            expect(item.y - item.height * 0.25, `${where} bottom`).toBeGreaterThanOrEqual(bottom - 2);
          }
        }
        await page.close();
      });
    }
  }

  it('renders every block exactly as tall as the paginator believed (lists included)', async () => {
    for (const template of TEMPLATES) {
      const page = await open(longResume(1, { template }));
      await page.evaluate(() => window.dispatchEvent(new Event('beforeprint')));
      const mismatches = await page.evaluate(() => {
        const bad = [];
        for (const article of document.querySelectorAll('article.page')) {
          const scale = article.getBoundingClientRect().height / article.offsetHeight;
          for (const el of article.querySelectorAll('[data-block]')) {
            const measured = window.__resume.measured(el.dataset.block);
            let expected = measured.height;
            if (measured.rows) {
              // a list fragment: the container's own space plus the rows this page shows
              const shown = [...el.querySelectorAll('[data-row]')];
              const first = Number(shown[0].dataset.row.split('.').pop());
              const chrome = Math.max(0, measured.height - measured.rows.reduce((a, b) => a + b, 0));
              expected = chrome + measured.rows.slice(first, first + shown.length).reduce((a, b) => a + b, 0);
            }
            const rendered = el.getBoundingClientRect().height / scale;
            if (Math.abs(rendered - expected) > 1) {
              bad.push(`${el.dataset.block}: rendered ${rendered.toFixed(1)}px, measured ${expected.toFixed(1)}px`);
            }
          }
        }
        return bad;
      });
      expect(mismatches, template).toEqual([]);
      await page.close();
    }
  });

  it('embeds real fonts (TrueType subsets, not Type 3) for every font pair', async () => {
    const expected = { inter: /Inter/, serif: /SourceSerif4/, plex: /IBMPlexSans/ };
    for (const [fontPair, family] of Object.entries(expected)) {
      const doc = longResume(1);
      doc['x-layout'].fontPair = fontPair;
      const page = await open(doc);
      const { pdf } = await print(page);
      expect(pdf.fonts.length, fontPair).toBeGreaterThan(0);
      expect(pdf.fonts.every((f) => !f.isType3 && !f.missingFile), `${fontPair} fonts`).toBe(true);
      expect(pdf.baseFonts.every((name) => /^[A-Z]{6}\+/.test(name)), `${fontPair} subset names`).toBe(true);
      expect(pdf.baseFonts.some((name) => family.test(name)), `${fontPair}: ${pdf.baseFonts}`).toBe(true);
      await page.close();
    }
  });

  it('reads in a sensible order for a tracking system: name, then summary, then experience', async () => {
    for (const template of TEMPLATES) {
      const page = await open(longResume(1, { template }));
      const { pdf } = await print(page);
      const text = squash(pdf.pages[0].text);
      expect(text.startsWith('AveryMorgan'), template).toBe(true);
      expect(text.indexOf('SUMMARY'), template).toBeGreaterThan(-1);
      expect(text.indexOf('SUMMARY')).toBeLessThan(text.indexOf('EXPERIENCE'));
      await page.close();
    }
  });

  it('names the PDF after the person', async () => {
    const page = await open(longResume(1));
    const { pdf } = await print(page);
    expect(pdf.title).toBe('Avery Morgan - Resume');
    await page.close();
  });

  it('keeps a very long unbroken word or URL inside the margins, with the same text', async () => {
    let doc = longResume(1, { template: 'classic' });
    doc = setPath(doc, 'work.0.highlights.0', `https://example.com/${'a'.repeat(300)} done`);
    doc = setPath(doc, 'work.0.highlights.1', 'x'.repeat(250));
    const page = await open(doc);
    const { shown, pdf } = await print(page);
    expect(pdf.pages).toHaveLength(shown.count);
    pdf.pages.forEach((p, i) => expect(squash(p.text)).toBe(squash(shown.texts[i])));
    const side = MARGIN_MM.x * PT_PER_MM;
    for (const p of pdf.pages) {
      for (const item of p.items.filter((it) => it.str.trim())) {
        expect(item.x + item.width).toBeLessThanOrEqual(p.width - side + 1);
      }
    }
    await page.close();
  });

  it('prints accented, Greek, Cyrillic and Japanese text exactly as shown', async () => {
    let doc = longResume(1, { template: 'classic' });
    doc = setPath(doc, 'basics.summary', 'データ分析と機械学習 — café résumé naïve — Ελληνικά — Привет мир — 日本語のテキスト');
    doc = setPath(doc, 'work.0.highlights.0', 'Œuvre complète: ß, ñ, ø, ł, ő, ž, ș, ț and the em dash — all in one line.');
    const page = await open(doc);
    const { shown, pdf } = await print(page);
    expect(pdf.pages).toHaveLength(shown.count);
    pdf.pages.forEach((p, i) => expect(squash(p.text)).toBe(squash(shown.texts[i])));
    await page.close();
  });

  it('survives emoji (printed by a system font): same page count, surrounding words intact', async () => {
    let doc = longResume(1, { template: 'classic' });
    doc = setPath(doc, 'basics.summary', 'Shipped 🚀 on time ✅ and loved it 😀 every time.');
    const page = await open(doc);
    const { shown, pdf } = await print(page);
    expect(pdf.pages).toHaveLength(shown.count);
    const text = squash(pdf.pages[0].text);
    expect(text).toContain('Shipped');
    expect(text).toContain('ontime');
    await page.close();
  });

  it('prints an edit made in the side panel', async () => {
    const page = await open(longResume(1, { template: 'sidebar' }));
    await page.evaluate(() => {
      const label = [...document.querySelectorAll('label')].find((l) => l.textContent.trim() === 'Name');
      const input = document.getElementById(label.htmlFor);
      input.value = 'Zed Zebra';
      input.dispatchEvent(new Event('input', { bubbles: true }));
    });
    const { shown, pdf } = await print(page);
    expect(squash(pdf.pages[0].text).startsWith('ZedZebra')).toBe(true);
    expect(shown.texts[0]).toContain('Zed Zebra');
    expect(pdf.title).toBe('Zed Zebra - Resume');
    await page.close();
  });

  it('"Fit to 3 pages" really produces a 3 page PDF', async () => {
    const page = await open(longResume(1, { template: 'classic' }));
    const before = await page.evaluate(() => Number(document.querySelector('[data-page-count]').dataset.pageCount));
    expect(before).toBeGreaterThan(3);
    await page.evaluate(() => {
      const label = [...document.querySelectorAll('label')].find((l) => l.textContent.includes('Fit to pages'));
      const input = document.getElementById(label.htmlFor);
      input.value = '3';
      input.dispatchEvent(new Event('input', { bubbles: true }));
      [...document.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Fit').click();
    });
    await page.waitForFunction(() => Number(document.querySelector('[data-page-count]').dataset.pageCount) <= 3);
    const { shown, pdf } = await print(page);
    expect(shown.count).toBeLessThanOrEqual(3);
    expect(pdf.pages).toHaveLength(shown.count);
    pdf.pages.forEach((p, i) => expect(squash(p.text)).toBe(squash(shown.texts[i])));
    await page.close();
  });
});

describe.skipIf(!chrome)('editing in a real browser', () => {
  it('keeps the caret in the field you are typing in when it moves to the next page', async () => {
    const page = await open(longResume(1, { template: 'classic' }));
    const path = await page.evaluate(() => {
      const fields = [...document.querySelector('article.page').querySelectorAll('[contenteditable][data-path]')];
      return fields.at(-1).dataset.path;
    });

    const where = () =>
      page.evaluate((p) => {
        const el = [...document.querySelectorAll('[contenteditable][data-path]')].find((e) => e.dataset.path === p);
        const selection = getSelection();
        const range = document.createRange();
        range.selectNodeContents(el);
        if (selection.anchorNode && el.contains(selection.anchorNode)) range.setEnd(selection.anchorNode, selection.anchorOffset);
        return {
          page: Number(el.closest('article').dataset.page),
          focused: document.activeElement === el,
          offset: range.toString().length,
          length: el.textContent.length,
        };
      }, path);

    await page.evaluate((p) => {
      const el = [...document.querySelectorAll('[contenteditable][data-path]')].find((e) => e.dataset.path === p);
      el.focus();
      const range = document.createRange();
      range.selectNodeContents(el);
      range.collapse(false);
      getSelection().removeAllRanges();
      getSelection().addRange(range);
    }, path);

    let moved = false;
    for (let i = 0; i < 25 && !moved; i++) {
      await page.keyboard.type(' lengthening this line until it has to move to the next page');
      await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
      moved = (await where()).page >= 1;
    }
    expect(moved, 'the field never moved to another page').toBe(true);

    const state = await where();
    expect(state.focused).toBe(true);
    expect(state.offset).toBe(state.length);

    await page.keyboard.type('END');
    const text = await page.evaluate((p) => [...document.querySelectorAll('[contenteditable][data-path]')].find((e) => e.dataset.path === p).textContent, path);
    expect(text.endsWith('END')).toBe(true);
    await page.close();
  });

  it('reorders a section when you drag its handle, and the page follows', async () => {
    const page = await open(longResume(1, { template: 'classic' }));
    const order = () =>
      page.evaluate(() => [...document.querySelectorAll('ol.sections > li')].map((li) => li.dataset.section));
    expect((await order()).indexOf('skills')).toBeGreaterThan((await order()).indexOf('work'));

    const centre = async (selector) => {
      const handle = await page.$(selector);
      await handle.scrollIntoView();
      const box = await handle.boundingBox();
      return { x: box.x + box.width / 2, y: box.y + box.height / 2, left: box.x, top: box.y };
    };
    const from = await centre('li[data-section="skills"] .handle');
    const to = await centre('li[data-section="work"]');
    await page.mouse.move(from.x, from.y);
    await page.mouse.down();
    await page.mouse.move(from.x + 10, from.y - 30, { steps: 6 });
    await page.mouse.move(to.left + 40, to.top + 4, { steps: 12 });
    await page.mouse.up();

    await page.waitForFunction(() => document.querySelector('ol.sections > li:nth-child(2)').dataset.section === 'skills');
    const headings = await page.evaluate(() =>
      [...document.querySelectorAll('article.page h2')].map((h) => h.textContent.trim()),
    );
    expect(headings.indexOf('Skills')).toBeLessThan(headings.indexOf('Experience'));
    await page.close();
  });

  it('keeps a comma you type in a skills list, without a space after it', async () => {
    const page = await open(longResume(1, { template: 'classic' }));
    const text = await page.evaluate(async () => {
      const el = document.querySelector('[contenteditable][data-path="skills.0.keywords"]');
      el.scrollIntoView();
      el.focus();
      return el.textContent;
    });
    expect(text).toContain('TypeScript');
    await page.keyboard.press('End');
    await page.keyboard.type(',Zig');
    await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
    const after = await page.evaluate(() => document.querySelector('[contenteditable][data-path="skills.0.keywords"]').textContent);
    expect(after.endsWith(',Zig')).toBe(true);
    await page.close();
  });

  it('pastes plain text only, and Enter does not split a single-line field', async () => {
    const page = await open(longResume(1, { template: 'classic' }));
    const result = await page.evaluate(() => {
      const el = document.querySelector('[contenteditable][data-path="basics.label"]');
      el.focus();
      const range = document.createRange();
      range.selectNodeContents(el);
      range.collapse(false);
      getSelection().removeAllRanges();
      getSelection().addRange(range);
      const data = new DataTransfer();
      data.setData('text/plain', ' and\nmore');
      data.setData('text/html', '<b>bold</b><script>1</script>');
      el.dispatchEvent(new ClipboardEvent('paste', { clipboardData: data, bubbles: true, cancelable: true }));
      return { text: el.textContent, children: el.children.length };
    });
    expect(result.text.endsWith('and more')).toBe(true);
    expect(result.children).toBe(0);
    await page.keyboard.press('Enter');
    const after = await page.evaluate(() => document.querySelector('[contenteditable][data-path="basics.label"]').textContent);
    expect(after.includes('\n')).toBe(false);
    await page.close();
  });

  it('lets Enter make a new line in the summary and keeps it in the PDF text', async () => {
    const page = await open(longResume(1, { template: 'classic' }));
    await page.evaluate(() => {
      const el = document.querySelector('[contenteditable][data-path="basics.summary"]');
      el.focus();
      const range = document.createRange();
      range.selectNodeContents(el);
      range.collapse(false);
      getSelection().removeAllRanges();
      getSelection().addRange(range);
    });
    await page.keyboard.press('Enter');
    await page.keyboard.type('Second paragraph');
    await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
    const { shown, pdf } = await print(page);
    expect(shown.texts[0]).toContain('Second paragraph');
    expect(squash(pdf.pages[0].text)).toContain('Secondparagraph');
    await page.close();
  });
});

describe.skipIf(!chrome)('accessibility and small screens', () => {
  const axePath = createRequire(import.meta.url).resolve('axe-core');

  for (const scheme of ['light', 'dark']) {
    it(`has no axe violations (${scheme} theme)`, async () => {
      const page = await open(longResume(1, { template: 'sidebar' }));
      await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: scheme }]);
      await page.addScriptTag({ path: axePath });
      // open every section so the forms are audited too
      await page.evaluate(() => document.querySelectorAll('.section-title').forEach((b) => b.click()));
      const violations = await page.evaluate(async () => {
        const result = await axe.run(document, { resultTypes: ['violations'] });
        return result.violations.map((v) => `${v.id}: ${v.help} (${v.nodes.length}) e.g. ${v.nodes[0]?.html.slice(0, 120)}`);
      });
      expect(violations).toEqual([]);
      await page.close();
    });
  }

  it('fits a phone: no sideways scrolling, the same pages, preview scaled to the screen', async () => {
    const desktop = await open(longResume(1, { template: 'classic' }));
    const expectedCount = await desktop.evaluate(() => Number(document.querySelector('[data-page-count]').dataset.pageCount));
    await desktop.close();

    const page = await open(longResume(1, { template: 'classic' }), { width: 390, height: 844 });
    const overflow = () => page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(await overflow()).toBeLessThanOrEqual(0);

    await page.evaluate(() => [...document.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Preview').click());
    await page.waitForFunction(() => document.querySelector('.stage').getBoundingClientRect().left >= 0);
    const frame = await page.evaluate(() => ({
      count: Number(document.querySelector('[data-page-count]').dataset.pageCount),
      width: document.querySelector('.pages-frame').getBoundingClientRect().width,
    }));
    expect(frame.count).toBe(expectedCount);
    expect(frame.width).toBeLessThanOrEqual(390);
    expect(await overflow()).toBeLessThanOrEqual(0);
    await page.close();
  });
});
