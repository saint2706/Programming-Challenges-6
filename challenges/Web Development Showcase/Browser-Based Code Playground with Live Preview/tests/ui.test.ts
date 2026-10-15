// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import { explainViolation } from '../src/app.ts';
import { ConsolePanel } from '../src/console.ts';
import { h } from '../src/dom.ts';
import { DEFAULT_TEMPLATE } from '../src/templates.ts';
import { loadSaved, save } from '../src/storage.ts';

describe('h() never turns text into markup', () => {
    test('text children and attribute values stay inert', () => {
        const el = h('p', { title: '"><img src=x onerror=alert(1)>' }, '<img src=x onerror=alert(1)>', h('b', {}, '<script>alert(1)</script>'));
        document.body.append(el);
        expect(document.querySelectorAll('img, script')).toHaveLength(0);
        expect(el.textContent).toBe('<img src=x onerror=alert(1)><script>alert(1)</script>');
        expect(el.getAttribute('title')).toBe('"><img src=x onerror=alert(1)>');
    });

    test('false and undefined attributes are left off, true is a bare attribute', () => {
        const el = h('input', { disabled: true, hidden: false, id: undefined, value: 3 });
        expect(el.hasAttribute('disabled')).toBe(true);
        expect(el.hasAttribute('hidden')).toBe(false);
        expect(el.hasAttribute('id')).toBe(false);
        expect(el.getAttribute('value')).toBe('3');
    });
});

describe('the console panel', () => {
    let el: HTMLElement;
    let panel: ConsolePanel;
    let jumps: [string, number, number][];

    /** Drawing is batched to one animation frame, so tests run it. */
    const frame = () => new Promise<void>((r) => requestAnimationFrame(() => r()));

    beforeEach(() => {
        document.body.innerHTML = '<div id="c"></div>';
        el = document.getElementById('c')!;
        jumps = [];
        panel = new ConsolePanel(el, (f, l, c) => jumps.push([f, l, c]));
    });

    test('messages are shown as text, whatever they contain', async () => {
        panel.add({ level: 'log', group: 0, text: '<img src=x onerror=alert(1)> & <b>bold</b>' });
        await frame();
        expect(el.querySelectorAll('img, b')).toHaveLength(0);
        expect(el.querySelector('.msg')!.textContent).toBe('<img src=x onerror=alert(1)> & <b>bold</b>');
    });

    test('a burst is drawn in one pass', async () => {
        const mutations: number[] = [];
        new MutationObserver((records) => mutations.push(records.length)).observe(el, { childList: true });
        for (let i = 0; i < 500; i++) panel.add({ level: 'log', group: 0, text: String(i) });
        await frame();
        await Promise.resolve();
        expect(el.childElementCount).toBe(500);
        expect(mutations.reduce((a, b) => a + b, 0)).toBeLessThanOrEqual(2);
    });

    test('entries waiting to be drawn are capped, so a page that posts faster than a frame can draw cannot pile up work', async () => {
        for (let i = 0; i < 50_000; i++) panel.add({ level: 'log', group: 0, text: String(i) });
        await frame();
        expect(el.childElementCount).toBeLessThanOrEqual(1000);
        expect(el.lastElementChild!.querySelector('.msg')!.textContent).toBe('49999'); // the newest is kept, the oldest dropped
    });

    test('only the last 1,000 are kept on screen', async () => {
        for (let i = 0; i < 1500; i++) panel.add({ level: 'log', group: 0, text: String(i) });
        await frame();
        expect(el.childElementCount).toBe(1000);
        expect(el.firstElementChild!.querySelector('.msg')!.textContent).toBe('500');
    });

    test('screen readers hear what kind of message it is, and errors link to the code', async () => {
        panel.add({ level: 'error', group: 0, text: 'bad', at: { file: 'main.ts', line: 3, column: 9 } });
        panel.add({ level: 'warn', group: 0, text: 'hmm' });
        panel.add({ level: 'log', group: 1, text: 'plain' });
        await frame();
        const rows = [...el.querySelectorAll('.entry')];
        expect(rows[0]!.querySelector('.sr-only')!.textContent).toBe('Error: ');
        expect(rows[1]!.querySelector('.sr-only')!.textContent).toBe('Warning: ');
        expect(rows[2]!.querySelector('.sr-only')).toBeNull();
        expect((rows[2] as HTMLElement).style.getPropertyValue('--g')).toBe('1');
        const link = rows[0]!.querySelector<HTMLButtonElement>('.where')!;
        expect(link.textContent).toBe('main.ts:3:9');
        link.click();
        expect(jumps).toEqual([['main.ts', 3, 9]]);
    });

    test('a stack is available on demand and clear empties everything', async () => {
        panel.add({ level: 'error', group: 0, text: 'x', stack: 'at main.ts:1:1' });
        await frame();
        expect(el.querySelector('details pre')!.textContent).toBe('at main.ts:1:1');
        const counts: number[][] = [];
        panel.onCounts = (w, e) => counts.push([w, e]);
        panel.clear();
        expect(el.childElementCount).toBe(0);
        expect(counts.at(-1)).toEqual([0, 0]);
    });

    test('counts of warnings and errors are reported, and the filter is a data attribute for CSS', async () => {
        const counts: number[][] = [];
        panel.onCounts = (w, e) => counts.push([w, e]);
        panel.add({ level: 'warn', group: 0, text: '1' });
        panel.add({ level: 'error', group: 0, text: '2' });
        panel.add({ level: 'error', group: 0, text: '3' });
        await frame();
        expect(counts.at(-1)).toEqual([1, 2]);
        panel.setFilter('error');
        expect(el.dataset.filter).toBe('error');
    });
});

describe('storage', () => {
    afterEach(() => {
        vi.restoreAllMocks();
        localStorage.clear();
    });

    test('saves and loads a project and the open file', () => {
        expect(save(DEFAULT_TEMPLATE.project, 'main.js')).toBe(true);
        const loaded = loadSaved();
        expect(loaded?.project).toEqual(DEFAULT_TEMPLATE.project);
        expect(loaded?.active).toBe('main.js');
    });

    test('an open file that no longer exists falls back to the first', () => {
        save(DEFAULT_TEMPLATE.project, 'gone.js');
        expect(loadSaved()?.active).toBe(DEFAULT_TEMPLATE.project.files[0]!.name);
    });

    test.each(['{not json', 'null', '[]', '{"project":{"files":[]}}', '{"project":{"files":[{"name":"../x","code":""}]}}', '{"project":5}'])('corrupt or hostile saved data %j is ignored', (raw) => {
        localStorage.setItem('playground:v1', raw);
        expect(loadSaved()).toBeNull();
    });

    test('blocked or full storage is reported, never thrown', () => {
        vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
            throw new DOMException('full', 'QuotaExceededError');
        });
        expect(save(DEFAULT_TEMPLATE.project, 'main.js')).toBe(false);
        vi.restoreAllMocks();
        vi.spyOn(window, 'localStorage', 'get').mockImplementation(() => {
            throw new DOMException('blocked', 'SecurityError');
        });
        expect(save(DEFAULT_TEMPLATE.project, 'main.js')).toBe(false);
        expect(loadSaved()).toBeNull();
    });
});

describe('explaining what the sandbox blocked', () => {
    test('an inline script or handler is explained in terms of what to do instead', () => {
        expect(explainViolation('script-src-elem', 'inline', false)).toMatch(/put it in main\.js/i);
        expect(explainViolation('script-src-attr', '', true)).toMatch(/addEventListener/);
    });

    test('a blocked network resource says whether the network is off or restricted', () => {
        expect(explainViolation('img-src', 'https://example.com/a.png', false)).toMatch(/Allow network/);
        expect(explainViolation('connect-src', 'https://example.com/api', true)).toMatch(/esm\.sh/);
        expect(explainViolation('img-src', '', false)).toContain('a resource');
    });
});
