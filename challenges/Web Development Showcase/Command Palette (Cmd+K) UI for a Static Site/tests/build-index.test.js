// @vitest-environment node
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { JSDOM } from 'jsdom';
import { describe, expect, it } from 'vitest';
import { buildIndex, indexPage } from '../scripts/build-index.mjs';

const root = fileURLToPath(new URL('..', import.meta.url));

const page = (main, head = '') => `<!doctype html><title>Fallback · Site</title>${head}<main>${main}</main>`;

describe('indexPage', () => {
    it('makes a page entry and a heading entry per h2/h3', () => {
        const items = indexPage(
            'guide.html',
            page(
                '<h1>The  Guide</h1><h2 id="a">First</h2><h3 id="b">Nested <em>one</em></h3>',
                '<meta name="description" content="About it"><meta name="keywords" content="how-to  tips">',
            ),
        );
        expect(items).toEqual([
            { id: 'p:guide.html', type: 'page', title: 'The Guide', description: 'About it', keywords: ['how-to', 'tips'], url: 'guide.html' },
            { id: 'h:guide.html#a', type: 'heading', title: 'First', hint: 'The Guide', url: 'guide.html#a' },
            { id: 'h:guide.html#b', type: 'heading', title: 'Nested one', hint: 'The Guide', url: 'guide.html#b' },
        ]);
    });

    it('falls back to the title without the site suffix, and omits empty fields', () => {
        const [entry] = indexPage('x.html', page(''));
        expect(entry).toEqual({ id: 'p:x.html', type: 'page', title: 'Fallback', url: 'x.html' });
    });

    it('skips the on-page contents box', () => {
        const items = indexPage('x.html', page('<nav class="toc"><h2 id="toc-title">On this page</h2></nav><h1>X</h1>'));
        expect(items).toHaveLength(1);
    });

    it('fails, naming the page and heading, when a heading has no id', () => {
        expect(() => indexPage('x.html', page('<h1>X</h1><h2>No anchor</h2>'))).toThrow(/x\.html.*"No anchor"/u);
    });

    it('does not index headings outside <main>', () => {
        const items = indexPage('x.html', '<title>X</title><header><h2 id="nav">Nav</h2></header><main><h1>X</h1></main>');
        expect(items).toHaveLength(1);
    });
});

describe('the demo site index', () => {
    const { items } = buildIndex(root);

    it('has every page, index.html first, with unique ids', () => {
        const files = readdirSync(root).filter((name) => name.endsWith('.html'));
        const pages = items.filter((item) => item.type === 'page');
        expect(pages.map((p) => p.url).sort()).toEqual([...files].sort());
        expect(pages[0].url).toBe('index.html');
        expect(new Set(items.map((i) => i.id)).size).toBe(items.length);
    });

    it('links every heading to an element that exists in its page', () => {
        for (const item of items.filter((i) => i.type === 'heading')) {
            const [file, id] = item.url.split('#');
            const { document } = new JSDOM(readFileSync(join(root, file), 'utf8')).window;
            expect(document.getElementById(id), item.url).not.toBeNull();
        }
    });

    it('is deterministic', () => {
        expect(JSON.stringify(buildIndex(root))).toBe(JSON.stringify(buildIndex(root)));
    });
});
