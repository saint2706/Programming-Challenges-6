import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { COLUMNS } from '../src/columns.js';
import { generateRows } from '../src/data.js';
import { mountDataTable } from '../src/grid.js';

const ROW_H = 36;
const GRID_H = 400;
const HEAD_H = 76;
const OVERSCAN = 6;

// jsdom has no layout: give the scroller and header the sizes the virtualizer measures.
beforeEach(() => {
  vi.useFakeTimers();
  // jsdom lacks Element#scrollTo, which the virtualizer uses for scrollToIndex.
  Element.prototype.scrollTo = function scrollTo(opts) {
    this.scrollTop = typeof opts === 'number' ? arguments[1] : opts.top;
    this.dispatchEvent(new Event('scroll'));
  };
  Object.defineProperty(HTMLElement.prototype, 'offsetHeight', {
    configurable: true,
    get() {
      if (this.classList.contains('grid')) return GRID_H;
      if (this.classList.contains('head')) return HEAD_H;
      return 0;
    },
  });
  Object.defineProperty(HTMLElement.prototype, 'offsetWidth', {
    configurable: true,
    get() {
      return this.classList.contains('grid') ? 1000 : 0;
    },
  });
  // The virtualizer clamps scrolling to scrollHeight - clientHeight; derive it from the spacer like a real layout would.
  Object.defineProperty(HTMLElement.prototype, 'scrollHeight', {
    configurable: true,
    get() {
      if (!this.classList.contains('grid')) return 0;
      const body = this.querySelector('.body');
      return HEAD_H + (parseFloat(body?.style.height) || 0);
    },
  });
  Object.defineProperty(HTMLElement.prototype, 'clientHeight', {
    configurable: true,
    get() {
      return this.classList.contains('grid') ? GRID_H : 0;
    },
  });
});

afterEach(() => {
  vi.useRealTimers();
  delete Element.prototype.scrollTo;
  delete HTMLElement.prototype.offsetHeight;
  delete HTMLElement.prototype.offsetWidth;
  delete HTMLElement.prototype.clientHeight;
  delete HTMLElement.prototype.scrollHeight;
  document.body.replaceChildren();
});

let root;
let table;
function mount(rows, opts = {}) {
  root = document.createElement('div');
  document.body.append(root);
  table = mountDataTable(root, { rows, debounceMs: 10, ...opts });
  return table;
}

const dataRows = () => [...root.querySelectorAll('.body [role=row]')];
const indices = () => dataRows().map((r) => Number(r.dataset.index));
const cellText = (row, colIndex) => row.children[colIndex].textContent;
const status = () => root.querySelector('[role=status]').textContent;
const scrollTo = (top) => {
  table.scroller.scrollTop = top;
  table.scroller.dispatchEvent(new Event('scroll'));
};
const type = (input, value) => {
  input.value = value;
  input.dispatchEvent(new Event('input', { bubbles: true }));
  vi.advanceTimersByTime(20);
};
const key = (k, init = {}) => {
  table.scroller.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true, ...init }));
  vi.advanceTimersByTime(50); // let scrollToIndex's reconcile loop settle
};
const readBlob = (blob) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsArrayBuffer(blob);
  });
const colIdx = (id) => COLUMNS.findIndex((c) => c.id === id);

describe('virtualization window', () => {
  it('renders only the visible rows plus overscan for a 5,000-row table', () => {
    mount(generateRows(5000));
    const visible = Math.ceil((GRID_H - HEAD_H) / ROW_H); // rows that fit under the sticky header
    const n = dataRows().length;
    expect(n).toBeGreaterThanOrEqual(visible);
    expect(n).toBeLessThanOrEqual(visible + OVERSCAN + 2);
    expect(n).toBeLessThan(40);
    expect(indices()[0]).toBe(0);
  });

  it('sizes the spacer for the full row count, so the scrollbar is honest', () => {
    mount(generateRows(5000));
    expect(root.querySelector('.body').style.height).toBe(`${5000 * ROW_H}px`);
  });

  it('positions rows absolutely by translateY(index * rowHeight)', () => {
    mount(generateRows(200));
    for (const r of dataRows()) expect(r.style.transform).toBe(`translateY(${Number(r.dataset.index) * ROW_H}px)`);
  });

  it('follows scrolling: the window moves and stays bounded', () => {
    mount(generateRows(5000));
    scrollTo(1000 * ROW_H);
    const idx = indices();
    // First visible row = floor((scrollTop - headerHeight) / rowHeight); the window starts OVERSCAN before it.
    const firstVisible = Math.floor((1000 * ROW_H - HEAD_H) / ROW_H);
    expect(idx[0]).toBe(firstVisible - OVERSCAN);
    // Last visible row = last one starting before scrollTop + viewport (less the header); OVERSCAN after it.
    const lastVisible = Math.ceil((1000 * ROW_H + GRID_H - HEAD_H) / ROW_H) - 1; // a row starting exactly on the edge is not visible
    expect(idx.at(-1)).toBe(lastVisible + OVERSCAN);
    expect(idx.length).toBeLessThan(40);
    expect(idx).toEqual([...idx].sort((a, b) => a - b));
  });

  it('reaches the very last row at the bottom', () => {
    mount(generateRows(300));
    scrollTo(300 * ROW_H);
    expect(indices().at(-1)).toBe(299);
  });

  it('renders the correct record at each window position', () => {
    const rows = generateRows(2000);
    mount(rows);
    scrollTo(500 * ROW_H);
    for (const r of dataRows()) expect(cellText(r, colIdx('id'))).toBe(String(rows[Number(r.dataset.index)].id));
  });
});

describe('ARIA structure', () => {
  it('exposes a grid with row/column counts and a header row', () => {
    mount(generateRows(50));
    const grid = table.scroller;
    expect(grid.getAttribute('role')).toBe('grid');
    expect(grid.getAttribute('aria-rowcount')).toBe('51');
    expect(grid.getAttribute('aria-colcount')).toBe(String(COLUMNS.length));
    const heads = root.querySelectorAll('[role=columnheader]');
    expect(heads).toHaveLength(COLUMNS.length);
    for (const h of heads) expect(h.getAttribute('aria-sort')).toBe('none');
    expect(dataRows()[0].getAttribute('aria-rowindex')).toBe('2');
  });

  it('announces the row count in a polite live region', () => {
    mount(generateRows(1234));
    const live = root.querySelector('[role=status]');
    expect(live.getAttribute('aria-live')).toBe('polite');
    expect(live.textContent).toMatch(/^Showing 1,234 of 1,234 rows/);
  });
});

describe('sorting through the UI', () => {
  it('a header click sorts ascending, reflects aria-sort, and resets scroll', () => {
    const rows = generateRows(1000);
    mount(rows);
    scrollTo(300 * ROW_H);
    root.querySelector('button[aria-label="Sort by Salary"]').click();
    expect(root.querySelectorAll('[role=columnheader]')[colIdx('salary')].getAttribute('aria-sort')).toBe('ascending');
    expect(table.scroller.scrollTop).toBe(0);
    const min = Math.min(...rows.filter((r) => r.salary !== null).map((r) => r.salary));
    expect(table.model.getRow(0).salary).toBe(min);
    expect(indices()[0]).toBe(0);
  });

  it('cycles asc -> desc -> off and updates the glyph', () => {
    mount(generateRows(300));
    const btn = root.querySelector('button[aria-label="Sort by Name"]');
    const head = root.querySelectorAll('[role=columnheader]')[colIdx('name')];
    btn.click();
    expect(head.querySelector('.glyph').textContent).toBe('▲');
    btn.click();
    expect(head.getAttribute('aria-sort')).toBe('descending');
    expect(head.querySelector('.glyph').textContent).toBe('▼');
    btn.click();
    expect(head.getAttribute('aria-sort')).toBe('none');
    expect(head.querySelector('.glyph').textContent).toBe('');
  });

  it('shift-click adds a secondary key and numbers the glyphs', () => {
    mount(generateRows(300));
    root.querySelector('button[aria-label="Sort by Department"]').click();
    root.querySelector('button[aria-label="Sort by Salary"]').dispatchEvent(new MouseEvent('click', { bubbles: true, shiftKey: true }));
    const heads = root.querySelectorAll('[role=columnheader]');
    expect(heads[colIdx('department')].querySelector('.glyph').textContent).toBe('▲1');
    expect(heads[colIdx('salary')].querySelector('.glyph').textContent).toBe('▲2');
  });
});

describe('filtering through the UI', () => {
  it('global search is debounced and updates the count and the rows', () => {
    const rows = generateRows(2000);
    mount(rows);
    const search = root.querySelector('input[type=search]');
    search.value = 'engineering berlin';
    search.dispatchEvent(new Event('input', { bubbles: true }));
    expect(status()).toMatch(/of 2,000/); // not yet applied: still debouncing
    expect(status()).toMatch(/^Showing 2,000/);
    vi.advanceTimersByTime(20);
    const expected = rows.filter((r) => JSON.stringify(r).toLowerCase().includes('engineering') && JSON.stringify(r).toLowerCase().includes('berlin')).length;
    expect(expected).toBeGreaterThan(0);
    expect(status()).toMatch(new RegExp(`^Showing ${expected.toLocaleString()} of 2,000`));
    expect(table.scroller.getAttribute('aria-rowcount')).toBe(String(expected + 1));
  });

  it('a numeric range filter on Salary applies, and a typo is flagged aria-invalid without blanking the table', () => {
    const rows = generateRows(1000);
    mount(rows);
    const input = root.querySelector('input[aria-label="Filter Salary"]');
    type(input, '100000..110000');
    const expected = rows.filter((r) => r.salary !== null && r.salary >= 100000 && r.salary <= 110000).length;
    expect(status()).toMatch(new RegExp(`^Showing ${expected.toLocaleString()} of 1,000`));
    expect(input.getAttribute('aria-invalid')).toBe('false');
    type(input, '100k');
    expect(input.getAttribute('aria-invalid')).toBe('true');
    expect(status()).toMatch(/^Showing 1,000 of 1,000/);
  });

  it('enum and boolean filters are selects', () => {
    const rows = generateRows(1000);
    mount(rows);
    const dept = root.querySelector('select[aria-label="Filter Department"]');
    dept.value = 'Legal';
    dept.dispatchEvent(new Event('change', { bubbles: true }));
    const legal = rows.filter((r) => r.department === 'Legal').length;
    expect(status()).toMatch(new RegExp(`^Showing ${legal} of 1,000`));
    expect(dataRows().every((r) => cellText(r, colIdx('department')) === 'Legal')).toBe(true);
    const active = root.querySelector('select[aria-label="Filter Active"]');
    active.value = 'false';
    active.dispatchEvent(new Event('change', { bubbles: true }));
    expect(table.model.visibleRows().every((r) => r.department === 'Legal' && r.active === false)).toBe(true);
  });

  it('shows an empty-state message when nothing matches', () => {
    mount(generateRows(100));
    type(root.querySelector('input[type=search]'), 'zzzz-nothing');
    expect(status()).toBe('No rows match. (100 total)');
    expect(dataRows()).toHaveLength(0);
    expect(table.scroller.getAttribute('aria-rowcount')).toBe('1');
    expect(table.scroller.hasAttribute('aria-activedescendant')).toBe(false);
  });

  it('Reset clears search, filters, sort and the aria-invalid flag', () => {
    mount(generateRows(500));
    type(root.querySelector('input[type=search]'), 'berlin');
    type(root.querySelector('input[aria-label="Filter Salary"]'), 'nope');
    root.querySelector('button[aria-label="Sort by Name"]').click();
    [...root.querySelectorAll('button')].find((b) => b.textContent === 'Reset').click();
    expect(status()).toMatch(/^Showing 500 of 500/);
    expect(root.querySelector('input[type=search]').value).toBe('');
    expect(root.querySelector('input[aria-label="Filter Salary"]').hasAttribute('aria-invalid')).toBe(false);
    expect(table.model.sorting).toEqual([]);
  });
});

describe('keyboard navigation (ARIA grid pattern)', () => {
  it('starts on the first cell and moves with the arrow keys', () => {
    mount(generateRows(500));
    const grid = table.scroller;
    expect(grid.getAttribute('aria-activedescendant')).toMatch(/-r0-c0$/);
    key('ArrowDown');
    expect(grid.getAttribute('aria-activedescendant')).toMatch(/-r1-c0$/);
    key('ArrowRight');
    key('ArrowRight');
    expect(grid.getAttribute('aria-activedescendant')).toMatch(/-r1-c2$/);
    key('ArrowUp');
    key('ArrowLeft');
    expect(grid.getAttribute('aria-activedescendant')).toMatch(/-r0-c1$/);
    expect(grid.querySelectorAll('.is-active')).toHaveLength(1);
  });

  it('clamps at the edges', () => {
    mount(generateRows(50));
    key('ArrowUp');
    key('ArrowLeft');
    expect(table.scroller.getAttribute('aria-activedescendant')).toMatch(/-r0-c0$/);
    key('End', { ctrlKey: true });
    key('ArrowDown');
    key('ArrowRight');
    expect(table.scroller.getAttribute('aria-activedescendant')).toMatch(new RegExp(`-r49-c${COLUMNS.length - 1}$`));
  });

  it('Ctrl+End / Ctrl+Home jump across the whole (virtualized) table and bring the cell into the DOM', () => {
    mount(generateRows(3000));
    key('End', { ctrlKey: true });
    const desc = table.scroller.getAttribute('aria-activedescendant');
    expect(desc).toMatch(new RegExp(`-r2999-c${COLUMNS.length - 1}$`));
    expect(document.getElementById(desc)).not.toBeNull();
    key('Home', { ctrlKey: true });
    expect(table.scroller.getAttribute('aria-activedescendant')).toMatch(/-r0-c0$/);
  });

  it('PageDown moves by roughly a screenful', () => {
    mount(generateRows(500));
    key('PageDown');
    const row = Number(/-r(\d+)-c/.exec(table.scroller.getAttribute('aria-activedescendant'))[1]);
    expect(row).toBeGreaterThanOrEqual(5);
    expect(row).toBeLessThanOrEqual(Math.ceil((GRID_H - HEAD_H) / ROW_H));
  });

  it('prevents default only for keys it handles, and ignores keys typed in a filter input', () => {
    mount(generateRows(50));
    const handled = new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true, cancelable: true });
    table.scroller.dispatchEvent(handled);
    expect(handled.defaultPrevented).toBe(true);
    const other = new KeyboardEvent('keydown', { key: 'a', bubbles: true, cancelable: true });
    table.scroller.dispatchEvent(other);
    expect(other.defaultPrevented).toBe(false);
    const input = root.querySelector('input[aria-label="Filter Name"]');
    const inInput = new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true, cancelable: true });
    input.dispatchEvent(inInput);
    expect(inInput.defaultPrevented).toBe(false);
    // Only the grid's own ArrowDown moved the active cell (r0 -> r1); the one from the input did not.
    expect(table.scroller.getAttribute('aria-activedescendant')).toMatch(/-r1-c0$/);
  });

  it('clicking a cell makes it the active cell', () => {
    mount(generateRows(100));
    const target = dataRows()[3].children[2];
    target.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(table.scroller.getAttribute('aria-activedescendant')).toBe(target.id);
    expect(target.classList.contains('is-active')).toBe(true);
  });

  it('keeps the active row in range after filtering shrinks the view', () => {
    mount(generateRows(500));
    key('End', { ctrlKey: true });
    type(root.querySelector('input[type=search]'), 'berlin');
    const row = Number(/-r(\d+)-c/.exec(table.scroller.getAttribute('aria-activedescendant'))[1]);
    expect(row).toBeLessThan(table.model.count);
  });
});

describe('untrusted data', () => {
  const hostile = [
    { id: 1, name: '<img src=x onerror="window.__pwned=1">', email: '"><script>window.__pwned=1</script>', city: '<b>bold</b>', department: 'Data', salary: 10, joined: '2020-01-01', active: true, notes: '<svg onload="window.__pwned=1">' },
  ];

  it('renders cell values as text, never as markup', () => {
    mount(hostile);
    expect(root.querySelector('.body img, .body script, .body svg, .body b')).toBeNull();
    expect(cellText(dataRows()[0], colIdx('name'))).toBe('<img src=x onerror="window.__pwned=1">');
    expect(window.__pwned).toBeUndefined();
  });

  it('exports hostile formulas neutralised', () => {
    mount([{ ...hostile[0], name: '=HYPERLINK("http://evil.invalid","x")', notes: '@SUM(1+1)' }]);
    const csv = table.exportCsv();
    expect(csv).toContain(`"'=HYPERLINK(""http://evil.invalid"",""x"")"`);
    expect(csv).toContain("'@SUM(1+1)");
  });
});

describe('CSV export', () => {
  it('exports the whole filtered + sorted view, not just the rendered window', () => {
    const rows = generateRows(2000);
    mount(rows);
    type(root.querySelector('input[type=search]'), 'engineering');
    root.querySelector('button[aria-label="Sort by Salary"]').click();
    root.querySelector('button[aria-label="Sort by Salary"]').click(); // descending
    const view = table.model.visibleRows();
    expect(view.length).toBeGreaterThan(dataRows().length); // more rows than are in the DOM
    const lines = table.exportCsv().split('\r\n');
    expect(lines.at(-1)).toBe('');
    const ids = lines.slice(1, -1).filter((l) => /^\d+,/.test(l)).map((l) => Number(l.split(',')[0]));
    expect(ids).toEqual(view.map((r) => r.id));
  });

  it('the Export button downloads a CSV blob honouring the BOM checkbox', async () => {
    const rows = generateRows(100);
    mount(rows);
    vi.useRealTimers(); // jsdom's FileReader needs real timers
    const blobs = [];
    const create = vi.fn((b) => (blobs.push(b), 'blob:x'));
    const revoke = vi.fn();
    vi.stubGlobal('URL', Object.assign(URL, { createObjectURL: create, revokeObjectURL: revoke }));
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    const button = [...root.querySelectorAll('button')].find((b) => b.textContent === 'Export CSV');
    button.click();
    expect(click).toHaveBeenCalledOnce();
    expect(blobs[0].type).toBe('text/csv;charset=utf-8');
    const bytes = new Uint8Array(await readBlob(blobs[0]));
    expect([...bytes.slice(0, 3)]).toEqual([0xef, 0xbb, 0xbf]); // BOM is on by default
    root.querySelector('input[type=checkbox]').checked = false;
    button.click();
    const plain = new Uint8Array(await readBlob(blobs[1]));
    expect(plain[0]).toBe('I'.charCodeAt(0)); // "ID,Name,..."
    click.mockRestore();
    vi.unstubAllGlobals();
  });
});

describe('lifecycle', () => {
  it('destroy() empties the root', () => {
    mount(generateRows(50));
    table.destroy();
    expect(root.childElementCount).toBe(0);
  });
});
