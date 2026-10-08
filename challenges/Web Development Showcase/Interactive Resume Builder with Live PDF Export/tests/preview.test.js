import { fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import Preview from '../src/components/Preview.svelte';
import { buildRegions } from '../src/lib/blocks.js';
import { createMeasurer } from '../src/lib/measure.js';
import { setLayout, setPath } from '../src/lib/model.js';
import { pageMetrics } from '../src/lib/page.js';
import { paginate } from '../src/lib/paginate.js';
import { emptyResume } from '../src/lib/schema.js';
import { longResume } from '../src/lib/sample-long.js';
import { seedResume } from '../src/lib/seed.js';

// jsdom has no layout engine, so give every block and every list row a fixed height.
const BLOCK_PX = 60;
const ROW_PX = 20;
let rectSpy;

beforeEach(() => {
  rectSpy = vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function rect() {
    const height = this.hasAttribute('data-row') ? ROW_PX : this.hasAttribute('data-block') ? BLOCK_PX : 0;
    return { height, width: 0, top: 0, left: 0, right: 0, bottom: height, x: 0, y: 0, toJSON() {} };
  });
});

afterEach(() => {
  rectSpy.mockRestore();
  vi.useRealTimers();
});

/** What the preview should produce, computed independently from the same fake heights. */
function expectedPageCount(doc) {
  const measurer = createMeasurer({
    measure: (blocks) =>
      blocks.map((b) => (b.rows ? { height: BLOCK_PX, rows: b.rows.map(() => ROW_PX) } : { height: BLOCK_PX })),
  });
  const resolved = measurer.resolve(buildRegions(doc), 'k');
  return paginate(resolved, pageMetrics(doc['x-layout'].pageSize).pagination).pageCount;
}

const pages = (container) => [...container.querySelectorAll('article.page')];

async function renderPreview(doc, props = {}) {
  const utils = render(Preview, { props: { doc, onedit: vi.fn(), ...props } });
  await waitFor(() => expect(utils.container.querySelector('[data-ready="true"]')).not.toBeNull());
  return utils;
}

describe('Preview', () => {
  it('paginates a long resume into the pages the paginator predicts', async () => {
    const doc = longResume();
    const { container } = await renderPreview(doc);
    expect(expectedPageCount(doc)).toBeGreaterThan(2);
    expect(pages(container)).toHaveLength(expectedPageCount(doc));
    expect(container.querySelector('[data-page-count]')).toHaveAttribute('data-page-count', String(expectedPageCount(doc)));
  });

  it('labels each page for assistive technology', async () => {
    const doc = longResume();
    const { container } = await renderPreview(doc);
    const total = pages(container).length;
    expect(screen.getByRole('article', { name: `Page 1 of ${total}` })).toBeInTheDocument();
    expect(screen.getByRole('article', { name: `Page ${total} of ${total}` })).toBeInTheDocument();
  });

  it('places every block and every bullet exactly once across the pages', async () => {
    const doc = longResume();
    const { container } = await renderPreview(doc);
    const rows = [...container.querySelectorAll('article [data-row]')].map((el) => el.getAttribute('data-row'));
    const expectedRows = doc.work.reduce((n, w) => n + w.highlights.length, 0) + doc.projects.reduce((n, p) => n + p.highlights.length, 0);
    expect(rows).toHaveLength(expectedRows);
    expect(new Set(rows).size).toBe(rows.length);

    const blocks = [...container.querySelectorAll('article [data-block]')].map((el) => el.getAttribute('data-block'));
    const whole = blocks.filter((id) => !id.endsWith('.bullets'));
    expect(new Set(whole).size).toBe(whole.length);
    expect(whole).toContain('header');
  });

  it('keeps the measuring tree out of the accessibility tree and out of the page count', async () => {
    const { container } = await renderPreview(longResume());
    const measure = container.querySelector('.measure');
    expect(measure).toHaveAttribute('aria-hidden', 'true');
    expect(measure.querySelector('[contenteditable]')).toBeNull();
    expect(measure.closest('article')).toBeNull();
  });

  it('renders one page for an empty resume, with placeholders to type into', async () => {
    const { container } = await renderPreview(emptyResume());
    expect(pages(container)).toHaveLength(1);
    expect(screen.getByRole('textbox', { name: 'Name' })).toHaveAttribute('data-placeholder', 'Your name');
  });

  it('renders the sidebar template as two regions on each page', async () => {
    const doc = setLayout(seedResume(), { template: 'sidebar' });
    const { container } = await renderPreview(doc);
    const first = pages(container)[0];
    expect(first.querySelector('.region-main')).not.toBeNull();
    expect(first.querySelector('.region-sidebar')).not.toBeNull();
    expect(first).toHaveClass('t-sidebar');
  });

  it('reports typing with the field path', async () => {
    const onedit = vi.fn();
    await renderPreview(seedResume(), { onedit });
    const name = screen.getByRole('textbox', { name: 'Name' });
    name.textContent = 'R. Agrawal';
    await fireEvent.input(name);
    expect(onedit).toHaveBeenCalledWith('basics.name', 'R. Agrawal');
  });

  it('coalesces a burst of edits into one repagination per frame', async () => {
    const doc = longResume();
    const { container, rerender } = await renderPreview(doc);
    vi.useFakeTimers({ toFake: ['requestAnimationFrame', 'cancelAnimationFrame'] });
    const before = rectSpy.mock.calls.length;
    let next = doc;
    for (let i = 0; i < 30; i++) {
      next = setPath(next, 'work.0.position', `Title ${i}`);
      await rerender({ doc: next });
    }
    expect(rectSpy.mock.calls.length).toBe(before);
    vi.advanceTimersByTime(50);
    expect(rectSpy.mock.calls.length).toBeGreaterThan(before);
    expect(container.querySelector('[data-block="work.0.header"]')).toHaveTextContent('Title 29');
  });

  it('measures only the block that changed', async () => {
    const doc = longResume();
    const { rerender } = await renderPreview(doc);
    const before = rectSpy.mock.calls.length;
    await rerender({ doc: setPath(doc, 'work.1.position', 'Principal Engineer') });
    await waitFor(() => expect(rectSpy.mock.calls.length).toBeGreaterThan(before));
    // one block measured: one rect read for it (it has no rows), not one per block
    expect(rectSpy.mock.calls.length - before).toBeLessThan(5);
  });

  it('re-measures everything when the template changes', async () => {
    const doc = longResume();
    const { container, rerender } = await renderPreview(doc);
    await rerender({ doc: setLayout(doc, { template: 'sidebar' }) });
    const sidebar = setLayout(doc, { template: 'sidebar' });
    await waitFor(() => expect(pages(container)).toHaveLength(expectedPageCount(sidebar)));
    expect(pages(container)[0]).toHaveClass('t-sidebar');
  });

  it('switches page size and keeps every page in that size', async () => {
    const doc = setLayout(longResume(), { pageSize: 'letter' });
    const { container } = await renderPreview(doc);
    expect(pages(container)).toHaveLength(expectedPageCount(doc));
    expect(pages(container)[0]).toHaveClass('size-letter');
    expect(document.head.querySelector('style[data-page-rule]')).toHaveTextContent('@page { size: Letter; margin: 0; }');
  });

  it('survives an hostile document: huge word, markup, every section hidden', async () => {
    let doc = setPath(longResume(), 'work.0.highlights.0', 'x'.repeat(20_000));
    doc = setPath(doc, 'basics.name', '<img src=x onerror=alert(1)>');
    for (const id of doc['x-layout'].sectionOrder) doc = { ...doc, 'x-layout': { ...doc['x-layout'], hidden: [...doc['x-layout'].hidden, id] } };
    const { container } = await renderPreview(doc);
    expect(container.querySelector('img')).toBeNull();
    expect(pages(container)).toHaveLength(1);
  });

  it('never measures a tree that is not laid out, so it cannot cache zero heights', async () => {
    const doc = longResume();
    const { container, component, rerender } = await renderPreview(doc);
    const shown = pages(container).length;
    const measure = container.querySelector('.measure');

    // e.g. a stylesheet hiding the measuring tree while printing
    measure.style.display = 'none';
    const grown = { ...doc, work: [...doc.work, ...doc.work, ...doc.work] };
    await rerender({ doc: grown });
    expect(() => component.flush()).not.toThrow();
    expect(pages(container)).toHaveLength(shown);

    // once it is laid out again the new content is measured properly: nothing was cached as zero
    measure.style.display = '';
    component.flush();
    expect(pages(container)).toHaveLength(expectedPageCount(grown));
  });

  it('does not repaginate between beforeprint and afterprint, but flush() still can', async () => {
    const doc = longResume();
    const { container, component, rerender } = await renderPreview(doc);
    const shown = pages(container).length;
    vi.useFakeTimers({ toFake: ['requestAnimationFrame', 'cancelAnimationFrame'] });

    window.dispatchEvent(new Event('beforeprint'));
    const grown = { ...doc, work: [...doc.work, ...doc.work, ...doc.work] };
    await rerender({ doc: grown });
    vi.advanceTimersByTime(100);
    expect(pages(container)).toHaveLength(shown);

    component.flush();
    expect(pages(container)).toHaveLength(expectedPageCount(grown));

    window.dispatchEvent(new Event('afterprint'));
    vi.advanceTimersByTime(100);
    expect(pages(container)).toHaveLength(expectedPageCount(grown));
  });

  it('flush() makes pagination current synchronously', async () => {
    const doc = longResume();
    const { container, component, rerender } = await renderPreview(doc);
    vi.useFakeTimers({ toFake: ['requestAnimationFrame', 'cancelAnimationFrame'] });
    const grown = { ...doc, work: [...doc.work, ...doc.work, ...doc.work] };
    await rerender({ doc: grown });
    expect(pages(container)).toHaveLength(expectedPageCount(doc));
    component.flush();
    expect(pages(container)).toHaveLength(expectedPageCount(grown));
  });

  it('pageCountAt() answers how many pages a scale would need without changing the preview', async () => {
    const doc = longResume();
    const { container, component } = await renderPreview(doc);
    const shown = pages(container).length;
    expect(component.pageCountAt(1)).toBe(shown);
    expect(pages(container)).toHaveLength(shown);
  });
});
