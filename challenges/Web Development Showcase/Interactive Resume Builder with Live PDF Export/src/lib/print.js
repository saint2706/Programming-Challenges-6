/**
 * Export to PDF is the browser's own print-to-PDF, so this module only makes sure the page is
 * ready and the saved file gets a sensible name: pagination is flushed, fonts are loaded, and the
 * document title (which browsers use as the default PDF file name) is "<name> - Resume" for the
 * duration of the print.
 */

const MAX_NAME = 80;

/** The document title to print under. */
export function documentTitle(name) {
  const clean = String(name ?? '')
    .replace(/[\\/:*?"<>|\u0000-\u001f]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, MAX_NAME)
    .trim();
  return clean ? `${clean} - Resume` : 'Resume';
}

/**
 * @param {object} options
 * @param {() => void} options.flush      make pagination current, synchronously
 * @param {() => string} options.getName  the person's name, read at print time
 */
export function createPrinter({
  flush,
  getName,
  fonts = globalThis.document?.fonts,
  win = globalThis.window,
  doc = globalThis.document,
}) {
  let original = null;

  const beforePrint = () => {
    flush();
    if (original === null) original = doc.title;
    doc.title = documentTitle(getName());
  };
  const afterPrint = () => {
    if (original === null) return;
    doc.title = original;
    original = null;
  };

  return {
    /** Wire up Ctrl+P / menu printing too. Returns a detach function. */
    attach() {
      win.addEventListener('beforeprint', beforePrint);
      win.addEventListener('afterprint', afterPrint);
      return () => {
        win.removeEventListener('beforeprint', beforePrint);
        win.removeEventListener('afterprint', afterPrint);
      };
    },

    /** The Export button: wait for fonts, flush, set the title, open the print dialog. */
    async print() {
      try {
        await fonts?.ready;
      } catch {
        // print with whatever fonts we have
      }
      beforePrint();
      win.print();
    },
  };
}
