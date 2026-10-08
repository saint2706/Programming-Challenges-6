import { fireEvent, render, screen, waitFor, within } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../src/App.svelte';
import { exportJson } from '../src/lib/io.js';
import { seedResume } from '../src/lib/seed.js';
import { CORRUPT_KEY, STORAGE_KEY } from '../src/lib/storage.js';

// jsdom has no layout engine: give every block and list row a fixed height.
let rectSpy;
beforeEach(() => {
  localStorage.clear();
  rectSpy = vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function rect() {
    const height = this.hasAttribute('data-row') ? 20 : this.hasAttribute('data-block') ? 60 : 0;
    return { height, width: 0, top: 0, left: 0, right: 0, bottom: height, x: 0, y: 0, toJSON() {} };
  });
});
afterEach(() => {
  rectSpy.mockRestore();
  vi.restoreAllMocks();
  vi.useRealTimers();
  localStorage.clear();
});

async function renderApp() {
  const utils = render(App);
  await waitFor(() => expect(utils.container.querySelector('[data-ready="true"]')).not.toBeNull());
  return { ...utils, user: userEvent.setup() };
}

const sectionOrder = (container) =>
  [...container.querySelectorAll('ol.sections > li')].map((li) => li.dataset.section);
const pageHeadings = (container) =>
  [...container.querySelectorAll('article.page h2')].map((h) => h.textContent.trim());
const statusText = () => screen.getAllByRole('status').map((el) => el.textContent.trim()).join(' | ');
// The name appears twice: typed on the page (a span) and in the side panel (an input).
const pageName = () => document.querySelector('article [data-path="basics.name"]');
const panelName = () => screen.getByLabelText('Name', { selector: 'input' });

async function startOver(user, label) {
  await user.click(screen.getByText('Start over'));
  await user.click(screen.getByRole('button', { name: label }));
}

describe('App: first load', () => {
  it('shows the starting resume, a page count and the editor', async () => {
    const { container } = await renderApp();
    expect(pageName()).toHaveTextContent('Rishabh Agrawal');
    expect(panelName()).toHaveValue('Rishabh Agrawal');
    expect(screen.getByText(/^\d+ pages?$/)).toBeInTheDocument();
    expect(container.querySelectorAll('article.page').length).toBeGreaterThan(0);
    expect(screen.getByRole('button', { name: 'Export PDF' })).toBeEnabled();
  });
});

describe('App: editing', () => {
  it('typing on the page updates the side panel and is saved when the page is hidden', async () => {
    await renderApp();
    const name = pageName();
    name.textContent = 'R. Agrawal';
    await fireEvent.input(name);
    expect(panelName()).toHaveValue('R. Agrawal');

    window.dispatchEvent(new Event('pagehide'));
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    expect(saved.doc.basics.name).toBe('R. Agrawal');
  });

  it('typing in the side panel updates the page', async () => {
    const { user } = await renderApp();
    const field = panelName();
    await user.clear(field);
    await user.type(field, 'Ada Lovelace');
    expect(pageName()).toHaveTextContent('Ada Lovelace');
  });

  it('keeps what you typed when you switch template and page size in the middle of editing', async () => {
    const { container, user } = await renderApp();
    const field = panelName();
    await user.clear(field);
    await user.type(field, 'Ada Lovelace');

    await user.click(screen.getByRole('radio', { name: /Compact/ }));
    await waitFor(() => expect(container.querySelector('article.page.t-compact')).not.toBeNull());
    await user.click(screen.getByRole('radio', { name: /US Letter/ }));
    await waitFor(() => expect(container.querySelector('article.page.size-letter')).not.toBeNull());

    expect(pageName()).toHaveTextContent('Ada Lovelace');
    expect(document.head.querySelector('style[data-page-rule]')).toHaveTextContent('Letter');
    expect(panelName()).toHaveValue('Ada Lovelace');
  });

  it('refuses a javascript: link and an invalid date, keeping the last good value', async () => {
    const { container, user } = await renderApp();
    const website = screen.getByLabelText('Website');
    await user.clear(website);
    await user.type(website, 'javascript:alert(1)');
    expect(screen.getByRole('alert')).toHaveTextContent('Use an http(s), mailto or tel link');
    expect(container.querySelector('[href^="javascript"]')).toBeNull();

    await user.clear(website);
    await user.type(website, 'https://example.org');
    expect(screen.queryByRole('alert')).toBeNull();
    expect(container.querySelector('article a[href="https://example.org"]')).not.toBeNull();

    await user.click(screen.getByRole('button', { name: /^Experience/ }));
    const start = screen.getAllByLabelText('Start')[0];
    const before = start.value;
    await user.clear(start);
    await user.type(start, 'soon');
    expect(screen.getByRole('alert')).toHaveTextContent('Use YYYY, YYYY-MM or YYYY-MM-DD');
    expect(start).toHaveAttribute('aria-invalid', 'true');
    await user.clear(start);
    await user.type(start, before);
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('adds and removes entries and announces it', async () => {
    const { user } = await renderApp();
    await user.click(screen.getByRole('button', { name: /^Awards/ }));
    await user.click(screen.getByRole('button', { name: 'Add award' }));
    expect(statusText()).toContain('New award added to Awards');
    expect(screen.getByRole('group', { name: 'New award 1' })).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Remove New award 1' }));
    expect(screen.queryByRole('group', { name: 'New award 1' })).toBeNull();
  });

  it('types a comma in the skills list in place without losing it', async () => {
    await renderApp();
    const skills = document.querySelector('article [data-path="skills.0.keywords"]');
    skills.focus();
    skills.textContent = 'Python,Go';
    await fireEvent.input(skills);
    expect(skills.textContent).toBe('Python,Go');
  });
});

describe('App: sections', () => {
  it('moves a section with Alt+ArrowUp, announces it and keeps focus on the handle', async () => {
    const { container, user } = await renderApp();
    const before = sectionOrder(container);
    const handle = screen.getByRole('button', { name: /Reorder Skills/ });
    handle.focus();
    await user.keyboard('{Alt>}{ArrowUp}{/Alt}');

    const after = sectionOrder(container);
    expect(after.indexOf('skills')).toBe(before.indexOf('skills') - 1);
    await waitFor(() => expect(statusText()).toContain('Skills moved up. Position 4 of 8.'));
    await waitFor(() => expect(document.activeElement).toBe(screen.getByRole('button', { name: /Reorder Skills/ })));
  });

  it('moves a section with the buttons and cannot move past the ends', async () => {
    const { container, user } = await renderApp();
    expect(screen.getByRole('button', { name: 'Move Summary up' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Move Awards down' })).toBeDisabled();
    await user.click(screen.getByRole('button', { name: 'Move Summary down' }));
    expect(sectionOrder(container).slice(0, 2)).toEqual(['work', 'summary']);
    await waitFor(() => expect(pageHeadings(container).slice(0, 2)).toEqual(['Experience', 'Summary']));
  });

  it('hides and shows a section on the page', async () => {
    const { container, user } = await renderApp();
    expect(pageHeadings(container)).toContain('Education');
    await user.click(screen.getByRole('checkbox', { name: /Show Education on the resume/ }));
    await waitFor(() => expect(pageHeadings(container)).not.toContain('Education'));
    expect(statusText()).toContain('Education hidden from the resume');
    await user.click(screen.getByRole('checkbox', { name: /Show Education on the resume/ }));
    await waitFor(() => expect(pageHeadings(container)).toContain('Education'));
  });

  it('renders a single page with only the header when every section is hidden', async () => {
    const { container, user } = await renderApp();
    for (const box of screen.getAllByRole('checkbox', { name: /on the resume/ })) await user.click(box);
    await waitFor(() => expect(pageHeadings(container)).toEqual([]));
    expect(container.querySelectorAll('article.page')).toHaveLength(1);
    expect(pageName()).toBeInTheDocument();
  });
});

describe('App: import, export and starting over', () => {
  async function upload(container, text, name = 'resume.json') {
    const input = container.querySelector('input[type="file"]');
    const file = new File([text], name, { type: 'application/json' });
    if (!file.text) file.text = async () => text;
    await fireEvent.change(input, { target: { files: [file] } });
  }

  it('imports a valid file and offers undo', async () => {
    const { container, user } = await renderApp();
    const doc = { ...seedResume(), basics: { ...seedResume().basics, name: 'Imported Person' } };
    await upload(container, exportJson(doc), 'imported.json');
    await waitFor(() => expect(panelName()).toHaveValue('Imported Person'));
    expect(container.querySelector('.notice')).toHaveTextContent('Imported imported.json.');
    await user.click(screen.getByRole('button', { name: 'Undo' }));
    expect(panelName()).toHaveValue('Rishabh Agrawal');
  });

  it('shows what is wrong with an invalid file and leaves the resume untouched', async () => {
    const { container } = await renderApp();
    await upload(container, JSON.stringify({ work: [{ startDate: 'soon' }] }));
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('work.0.startDate'));
    expect(panelName()).toHaveValue('Rishabh Agrawal');

    await upload(container, '{ nope');
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('not valid JSON'));
    expect(panelName()).toHaveValue('Rishabh Agrawal');
  });

  it('rejects a hostile 10,000-entry file without freezing or applying it', async () => {
    const { container } = await renderApp();
    const work = Array.from({ length: 10_000 }, (_, i) => ({ name: `Company ${i}` }));
    await upload(container, JSON.stringify({ basics: { name: 'Hostile' }, work }));
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('work'));
    expect(panelName()).toHaveValue('Rishabh Agrawal');
  });

  it('starts a blank resume, keeps the chosen design, and can undo', async () => {
    const { container, user } = await renderApp();
    await user.click(screen.getByRole('radio', { name: /Compact/ }));
    await startOver(user, 'Blank');
    expect(panelName()).toHaveValue('');
    expect(screen.getByRole('radio', { name: /Compact/ })).toBeChecked();
    await waitFor(() => expect(container.querySelectorAll('article.page')).toHaveLength(1));
    await user.click(screen.getByRole('button', { name: 'Undo' }));
    expect(panelName()).toHaveValue('Rishabh Agrawal');
  });

  it('dismisses a message', async () => {
    const { container, user } = await renderApp();
    await startOver(user, 'Blank');
    expect(container.querySelector('.notice')).toHaveTextContent('Started a blank resume.');
    await user.click(screen.getByRole('button', { name: 'Dismiss message' }));
    expect(container.querySelector('.notice')).toBeNull();
  });
});

describe('App: storage', () => {
  it('works with storage disabled and says so', async () => {
    vi.spyOn(window, 'localStorage', 'get').mockImplementation(() => {
      throw new DOMException('denied', 'SecurityError');
    });
    const { user } = await renderApp();
    expect(screen.getByText(/Browser storage is unavailable/)).toBeInTheDocument();
    const field = panelName();
    await user.clear(field);
    await user.type(field, 'Still Works');
    expect(pageName()).toHaveTextContent('Still Works');
    window.dispatchEvent(new Event('pagehide'));
  });

  it('recovers from corrupt saved data, keeping a backup', async () => {
    localStorage.setItem(STORAGE_KEY, '{definitely not json');
    await renderApp();
    expect(screen.getByRole('alert')).toHaveTextContent('could not be read');
    expect(localStorage.getItem(CORRUPT_KEY)).toBe('{definitely not json');
    expect(panelName()).toHaveValue('Rishabh Agrawal');
  });

  it('restores the saved resume on the next visit', async () => {
    const { unmount } = await renderApp();
    const field = panelName();
    await userEvent.setup().clear(field);
    await userEvent.setup().type(field, 'Saved Person');
    window.dispatchEvent(new Event('pagehide'));
    unmount();
    await renderApp();
    expect(panelName()).toHaveValue('Saved Person');
  });
});

describe('App: fit to pages', () => {
  it('reports when a resume cannot fit, and fills the page when it can', async () => {
    const { user } = await renderApp();
    await startOver(user, 'Long example (3+ pages)');
    await waitFor(() => expect(screen.getByText(/^[3-9] pages$/)).toBeInTheDocument());

    const target = screen.getByLabelText(/Fit to pages/);
    await user.clear(target);
    await user.type(target, '1');
    await user.click(screen.getByRole('button', { name: 'Fit' }));
    expect(screen.getAllByRole('status').map((el) => el.textContent).join(' ')).toMatch(/Cannot fit in 1 page even at 82%: it needs \d+/);

    await user.clear(target);
    await user.type(target, '20');
    await user.click(screen.getByRole('button', { name: 'Fit' }));
    expect(screen.getAllByRole('status').map((el) => el.textContent).join(' ')).toMatch(/Fits in 20 pages at 115% text size/);
    expect(screen.getByLabelText('Text size')).toHaveValue('1.15');
  });

  it('asks for a whole number of pages', async () => {
    const { user } = await renderApp();
    const target = screen.getByLabelText(/Fit to pages/);
    await user.clear(target);
    await user.click(screen.getByRole('button', { name: 'Fit' }));
    expect(screen.getAllByRole('status').map((el) => el.textContent).join(' ')).toContain('Enter a whole number of pages');
  });
});

describe('App: export', () => {
  it('flushes, names the document after the person, prints, then restores the title', async () => {
    const { user } = await renderApp();
    const titles = [];
    window.print = vi.fn(() => titles.push(document.title));
    const original = document.title;
    await user.click(screen.getByRole('button', { name: 'Export PDF' }));
    await waitFor(() => expect(window.print).toHaveBeenCalledTimes(1));
    expect(titles).toEqual(['Rishabh Agrawal - Resume']);
    window.dispatchEvent(new Event('afterprint'));
    expect(document.title).toBe(original);
  });

  it('prints a resume with no name under a plain title', async () => {
    const { user } = await renderApp();
    await startOver(user, 'Blank');
    const titles = [];
    window.print = vi.fn(() => titles.push(document.title));
    await user.click(screen.getByRole('button', { name: 'Export PDF' }));
    await waitFor(() => expect(window.print).toHaveBeenCalled());
    expect(titles).toEqual(['Resume']);
  });
});

describe('App: small screens', () => {
  it('switches between the editor and the preview', async () => {
    const { container, user } = await renderApp();
    const app = container.querySelector('.app');
    expect(app).toHaveAttribute('data-view', 'edit');
    expect(screen.getByRole('button', { name: 'Edit' })).toHaveAttribute('aria-pressed', 'true');
    await user.click(screen.getByRole('button', { name: 'Preview' }));
    expect(app).toHaveAttribute('data-view', 'preview');
    expect(screen.getByRole('button', { name: 'Preview' })).toHaveAttribute('aria-pressed', 'true');
    // the preview stays mounted (it measures itself) whichever view is showing
    expect(within(container).getByRole('region', { name: 'Resume preview' })).toBeInTheDocument();
  });
});
