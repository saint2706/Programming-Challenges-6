import { getDocument } from 'pdfjs-dist/legacy/build/pdf.mjs';

/** Remove every whitespace character: PDF text runs and DOM text differ only in spacing. */
export const squash = (text) => text.replace(/\s+/g, '');

/**
 * Read a PDF the way an applicant tracking system would: page sizes, the text of each page with
 * the position of every run, which fonts were used and whether each is a real embedded font.
 */
export async function readPdf(bytes) {
  // Font names live in plain objects in Chrome's PDFs: `/BaseFont /AAAAAA+Inter-Bold`. Read them
  // first: pdf.js takes ownership of the buffer it is given and leaves it empty.
  const raw = Buffer.from(bytes).toString('latin1');
  const baseFonts = [...new Set([...raw.matchAll(/\/BaseFont\s*\/([^\s/>\]]+)/g)].map((m) => m[1]))];

  const task = getDocument({ data: new Uint8Array(bytes), isEvalSupported: false, useSystemFonts: false });
  const doc = await task.promise;
  const pages = [];
  const fonts = new Map();

  for (let number = 1; number <= doc.numPages; number++) {
    const page = await doc.getPage(number);
    const viewport = page.getViewport({ scale: 1 });
    const content = await page.getTextContent();
    await page.getOperatorList(); // makes pdf.js load the page's fonts
    const items = content.items
      .filter((item) => 'str' in item)
      .map((item) => ({
        str: item.str,
        x: item.transform[4],
        y: item.transform[5],
        width: item.width,
        height: item.height,
        font: item.fontName,
      }));
    for (const id of new Set(items.map((item) => item.font).filter(Boolean))) {
      if (!page.commonObjs.has(id)) continue;
      const font = page.commonObjs.get(id);
      fonts.set(`${number}:${id}`, { isType3: Boolean(font.isType3Font), missingFile: Boolean(font.missingFile) });
    }
    pages.push({ width: viewport.width, height: viewport.height, items, text: items.map((item) => item.str).join('') });
  }

  const metadata = await doc.getMetadata();
  await task.destroy();

  return { pages, fonts: [...fonts.values()], baseFonts, title: metadata.info?.Title ?? '' };
}
