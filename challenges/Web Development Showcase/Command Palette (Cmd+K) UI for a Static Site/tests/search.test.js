import { describe, expect, it } from 'vitest';
import { createSearch, tokenize } from '../src/search.js';

const items = [
    { id: 'a:theme', type: 'action', title: 'Toggle color theme', keywords: ['dark', 'light'] },
    { id: 'a:top', type: 'action', title: 'Scroll to top' },
    { id: 'p:start', type: 'page', title: 'Getting started', hint: 'Set the palette up', url: 'start.html' },
    { id: 'p:keys', type: 'page', title: 'Keyboard shortcuts', keywords: ['hotkeys'], url: 'keys.html' },
    { id: 'h:start#install', type: 'heading', title: 'Install', hint: 'Getting started', url: 'start.html#install' },
    { id: 'h:keys#install', type: 'heading', title: 'Installing keys', hint: 'Keyboard shortcuts', url: 'keys.html#installing' },
    { id: 'h:start#gaps', type: 'heading', title: 'Gaps', hint: 'Getting started', url: 'start.html#gaps' },
];
const search = createSearch(items);
const ids = (query) => search.rank(query).map((hit) => hit.item.id);

describe('rank', () => {
    it('finds items by a fuzzy fragment of the title', () => {
        expect(ids('gs')[0]).toBe('p:start');
        expect(ids('ksh')[0]).toBe('p:keys');
    });

    it('puts the exact title before a longer one that contains it', () => {
        expect(ids('install').slice(0, 2)).toEqual(['h:start#install', 'h:keys#install']);
    });

    it('requires every word, each in the title, hint or keywords', () => {
        expect(ids('gaps getting')).toEqual(['h:start#gaps']);
        expect(ids('gaps keyboard')).toEqual([]);
    });

    it('matches a keyword that is not in the title', () => {
        expect(ids('hotkeys')).toEqual(['p:keys']);
        expect(ids('dark')).toEqual(['a:theme']);
    });

    it('scores a title match above a hint match for the same word', () => {
        // "started" is the title of the page and only the hint of its headings
        const order = ids('started');
        expect(order[0]).toBe('p:start');
        expect(order).toContain('h:start#install');
    });

    it('only highlights title letters', () => {
        const hit = search.rank('started').find((h) => h.item.id === 'h:start#install');
        expect(hit.positions).toEqual([]);
        expect(search.rank('inst')[0].positions).toEqual([0, 1, 2, 3]);
    });

    it('returns nothing for a blank query or no match', () => {
        expect(ids('')).toEqual([]);
        expect(ids('   ')).toEqual([]);
        expect(ids('zzzq')).toEqual([]);
    });

    it('honours a limit', () => {
        expect(search.rank('t', 2)).toHaveLength(2);
    });

    it('breaks score ties by shorter title, then original order', () => {
        const tie = createSearch([
            { id: '1', type: 'page', title: 'Notes extra' },
            { id: '2', type: 'page', title: 'Notes' },
            { id: '3', type: 'page', title: 'Notes' },
        ]);
        expect(tie.rank('notes').map((h) => h.item.id)).toEqual(['2', '3', '1']);
    });
});

describe('tokenize', () => {
    it('splits on whitespace and drops empties', () => {
        expect(tokenize('  a   b ')).toHaveLength(2);
        expect(tokenize('')).toEqual([]);
    });

    it('caps the number of words and their length', () => {
        expect(tokenize('a b c d e f g h')).toHaveLength(6);
        expect(tokenize('x'.repeat(100))[0]).toHaveLength(32);
    });
});

describe('search sections', () => {
    it('groups by type and orders the groups by their best score', () => {
        expect(search.search('install').map((s) => s.type)).toEqual(['heading']);
        const mixed = search.search('k');
        expect(mixed.map((s) => s.type)).toContain('page');
        const best = mixed.map((s) => s.results[0].score);
        expect(best).toEqual([...best].sort((a, b) => b - a));
    });

    it('caps each section', () => {
        const many = createSearch(
            Array.from({ length: 30 }, (_, i) => ({ id: `${i}`, type: 'heading', title: `Heading ${i}` })),
        );
        expect(many.search('head')[0].results).toHaveLength(10);
    });

    it('with no query shows recents first, then actions and pages, never headings', () => {
        const sections = search.search('', { recentIds: ['h:start#gaps', 'p:keys', 'gone'] });
        expect(sections.map((s) => s.type)).toEqual(['recent', 'action', 'page']);
        expect(sections[0].results.map((r) => r.item.id)).toEqual(['h:start#gaps', 'p:keys']);
        // a recent page is not repeated in Pages
        expect(sections[2].results.map((r) => r.item.id)).toEqual(['p:start']);
    });
});
