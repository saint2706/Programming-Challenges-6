import { describe, expect, it, vi } from 'vitest';
import { createPrinter, documentTitle } from '../src/lib/print.js';

describe('documentTitle', () => {
  it('names the PDF after the person', () => {
    expect(documentTitle('Ada Lovelace')).toBe('Ada Lovelace - Resume');
  });

  it('falls back to a plain title for an empty or blank name', () => {
    expect(documentTitle('')).toBe('Resume');
    expect(documentTitle('   ')).toBe('Resume');
    expect(documentTitle(undefined)).toBe('Resume');
  });

  it('strips characters that make a bad file name', () => {
    expect(documentTitle('A/B\\C:D*E?"F<G>H|I')).toBe('A B C D E F G H I - Resume');
    expect(documentTitle('x'.repeat(200)).length).toBeLessThanOrEqual(80 + ' - Resume'.length);
  });
});

function setup({ name = 'Ada' } = {}) {
  const order = [];
  const win = new EventTarget();
  win.print = vi.fn(() => order.push(`print:${doc.title}`));
  const doc = { title: 'Resume builder' };
  const fonts = { ready: Promise.resolve() };
  const flush = vi.fn(() => order.push('flush'));
  const printer = createPrinter({ flush, getName: () => name, fonts, win, doc });
  return { printer, win, doc, flush, order };
}

describe('createPrinter', () => {
  it('flushes pagination, sets the title, then prints', async () => {
    const { printer, win, flush, order } = setup();
    await printer.print();
    expect(flush).toHaveBeenCalledTimes(1);
    expect(win.print).toHaveBeenCalledTimes(1);
    expect(order).toEqual(['flush', 'print:Ada - Resume']);
  });

  it('restores the title after printing', async () => {
    const { printer, win, doc } = setup();
    printer.attach();
    await printer.print();
    expect(doc.title).toBe('Ada - Resume');
    win.dispatchEvent(new Event('afterprint'));
    expect(doc.title).toBe('Resume builder');
  });

  it('also flushes and titles when printing starts from the browser (Ctrl+P)', () => {
    const { printer, win, doc, flush } = setup();
    printer.attach();
    win.dispatchEvent(new Event('beforeprint'));
    expect(flush).toHaveBeenCalledTimes(1);
    expect(doc.title).toBe('Ada - Resume');
    win.dispatchEvent(new Event('afterprint'));
    expect(doc.title).toBe('Resume builder');
  });

  it('prints with an empty name', async () => {
    const { printer, doc, order } = setup({ name: '' });
    await printer.print();
    expect(order.at(-1)).toBe('print:Resume');
    expect(doc.title).toBe('Resume');
  });

  it('remembers the original title across repeated beforeprint events', () => {
    const { printer, win, doc } = setup();
    printer.attach();
    win.dispatchEvent(new Event('beforeprint'));
    win.dispatchEvent(new Event('beforeprint'));
    win.dispatchEvent(new Event('afterprint'));
    expect(doc.title).toBe('Resume builder');
  });

  it('stops listening once detached', () => {
    const { printer, win, flush } = setup();
    const detach = printer.attach();
    detach();
    win.dispatchEvent(new Event('beforeprint'));
    expect(flush).not.toHaveBeenCalled();
  });

  it('still prints when fonts never report ready', async () => {
    const { win, doc } = setup();
    const printer = createPrinter({
      flush: () => {},
      getName: () => 'Ada',
      fonts: { ready: Promise.reject(new Error('no')) },
      win,
      doc,
    });
    await printer.print();
    expect(win.print).toHaveBeenCalled();
  });
});
