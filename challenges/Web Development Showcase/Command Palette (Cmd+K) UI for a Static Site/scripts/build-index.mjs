import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { basename, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { JSDOM } from 'jsdom';

/**
 * The build-time page index: every top-level `.html` file becomes a `page` entry (title,
 * description, optional `<meta name="keywords">`) plus a `heading` entry for each `h2`/`h3` inside
 * `<main>`. Output is deterministic (no timestamp), so it diffs cleanly and tests can compare it.
 *
 * A heading without an `id` is an error, not a skip: a result that cannot link anywhere is a bug in
 * the page, and silently dropping it would hide the section from search.
 */

const SITE_SUFFIX = / · [^·]+$/u;
const squash = (text) => text.replace(/\s+/gu, ' ').trim();

/** Index one HTML string. `file` is its name relative to the site root, e.g. `matching.html`. */
export function indexPage(file, source) {
    const { document } = new JSDOM(source).window;
    const title = squash(document.querySelector('h1')?.textContent ?? '') || squash(document.title.replace(SITE_SUFFIX, ''));
    if (!title) throw new Error(`${file}: no <h1> or <title>`);
    const description = squash(document.querySelector('meta[name="description"]')?.content ?? '');
    const keywords = squash(document.querySelector('meta[name="keywords"]')?.content ?? '')
        .split(' ')
        .filter(Boolean);

    const items = [
        {
            id: `p:${file}`,
            type: 'page',
            title,
            ...(description && { description }),
            ...(keywords.length && { keywords }),
            url: file,
        },
    ];
    const missing = [];
    for (const heading of document.querySelectorAll('main h2, main h3')) {
        // The "On this page" box (.toc) is page chrome, not a section.
        if (heading.closest('.toc')) continue;
        const text = squash(heading.textContent);
        if (!heading.id) {
            missing.push(text);
            continue;
        }
        items.push({ id: `h:${file}#${heading.id}`, type: 'heading', title: text, hint: title, url: `${file}#${heading.id}` });
    }
    if (missing.length) throw new Error(`${file}: headings without an id: ${missing.map((t) => `"${t}"`).join(', ')}`);
    return items;
}

/** Index every page in `dir`: `index.html` first, the rest alphabetically. */
export function buildIndex(dir) {
    const files = readdirSync(dir)
        .filter((name) => name.endsWith('.html'))
        .sort((a, b) => (a === 'index.html' ? -1 : b === 'index.html' ? 1 : a.localeCompare(b)));
    const items = files.flatMap((file) => indexPage(file, readFileSync(join(dir, file), 'utf8')));
    const ids = new Set();
    for (const { id } of items) {
        if (ids.has(id)) throw new Error(`duplicate id in the index: ${id}`);
        ids.add(id);
    }
    return { v: 1, items };
}

if (process.argv[1] && basename(process.argv[1]) === 'build-index.mjs') {
    const root = fileURLToPath(new URL('..', import.meta.url));
    const out = join(root, 'src', 'generated');
    mkdirSync(out, { recursive: true });
    const index = buildIndex(root);
    writeFileSync(join(out, 'search-index.json'), `${JSON.stringify(index)}\n`);
    const pages = index.items.filter((item) => item.type === 'page').length;
    console.log(`search index: ${pages} pages, ${index.items.length - pages} headings`);
}
