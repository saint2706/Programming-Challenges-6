import { matchPrepared, preparePattern, prepareText, MAX_PATTERN } from './fuzzy.js';

/**
 * Query -> ranked, sectioned results over a list of items:
 * `{ id, type: 'page' | 'heading' | 'action', title, hint?, description?, keywords?, url? }`.
 *
 * A query is split on spaces into tokens and an item must match every token (fzf's AND), each in
 * its title, its hint (the page a heading lives on) or its keywords. A title match counts in full,
 * a keyword match at 60% and a hint match at 50%, so "install" ranks the heading called Install
 * above a heading that merely sits on a page called Installing. Only title matches are highlighted.
 * A `description` is shown under the result but never searched: long prose fuzzily matches almost
 * any short query.
 */

const WEIGHT = { title: 1, keywords: 0.6, hint: 0.5 };
const MAX_TOKENS = 6;

/** Sections in the order they show when they score equally, with how many results each may show. */
export const SECTIONS = [
    { type: 'action', label: 'Actions', cap: 6 },
    { type: 'page', label: 'Pages', cap: 8 },
    { type: 'heading', label: 'Headings', cap: 10 },
];

export function tokenize(query) {
    return query
        .trim()
        .split(/\s+/u)
        .filter(Boolean)
        .slice(0, MAX_TOKENS)
        .map((token) => Array.from(token).slice(0, MAX_PATTERN).join(''))
        .map(preparePattern)
        .filter(Boolean);
}

function prepareItem(item) {
    return {
        item,
        title: prepareText(item.title),
        hint: item.hint ? prepareText(item.hint) : null,
        keywords: item.keywords?.length ? prepareText(item.keywords.join(' ')) : null,
    };
}

/** One item against all tokens: `{ score, positions }` or `null` if any token finds nothing. */
function matchItem(entry, tokens) {
    let total = 0;
    const positions = [];
    for (const token of tokens) {
        const inTitle = matchPrepared(token, entry.title);
        const inHint = entry.hint ? matchPrepared(token, entry.hint) : null;
        const inKeywords = entry.keywords ? matchPrepared(token, entry.keywords) : null;
        const titleScore = inTitle ? inTitle.score * WEIGHT.title : -1;
        const hintScore = inHint ? inHint.score * WEIGHT.hint : -1;
        const keywordScore = inKeywords ? inKeywords.score * WEIGHT.keywords : -1;
        const best = Math.max(titleScore, hintScore, keywordScore);
        if (best < 0) return null;
        total += best;
        if (inTitle && titleScore === best) positions.push(...inTitle.positions);
    }
    return { score: total, positions: [...new Set(positions)].sort((a, b) => a - b) };
}

/**
 * Prepare the items once and get back `search(query)`. `rank(query)` is the flat ranked list (what
 * the benchmark measures); `search` groups it into sections for the palette.
 */
export function createSearch(items) {
    const entries = items.map(prepareItem);

    function rank(query, limit = Infinity) {
        const tokens = tokenize(query);
        if (!tokens.length) return [];
        const hits = [];
        for (let index = 0; index < entries.length; index += 1) {
            const match = matchItem(entries[index], tokens);
            if (match) hits.push({ item: entries[index].item, index, ...match });
        }
        // Higher score first; then the shorter title (a closer match), then the original order.
        hits.sort(
            (a, b) =>
                b.score - a.score ||
                a.item.title.length - b.item.title.length ||
                a.index - b.index,
        );
        return hits.length > limit ? hits.slice(0, limit) : hits;
    }

    /**
     * Sections for the palette. With a query: each section's best matches, sections ordered by their
     * best score. With none: `recentIds` first, then actions and pages (no headings: too many).
     */
    function search(query, { recentIds = [] } = {}) {
        if (!tokenize(query).length) return suggestions(entries, recentIds);
        const hits = rank(query);
        const sections = [];
        for (const { type, label, cap } of SECTIONS) {
            const own = hits.filter((hit) => hit.item.type === type).slice(0, cap);
            if (own.length) sections.push({ type, label, results: own });
        }
        // Stable: sections that tie keep the SECTIONS order.
        sections.sort((a, b) => b.results[0].score - a.results[0].score);
        return sections;
    }

    return { rank, search, size: entries.length };
}

function suggestions(entries, recentIds) {
    const byId = new Map(entries.map((entry) => [entry.item.id, entry.item]));
    const plain = (item) => ({ item, score: 0, positions: [] });
    const sections = [];
    const recent = recentIds.map((id) => byId.get(id)).filter(Boolean);
    if (recent.length) sections.push({ type: 'recent', label: 'Recent', results: recent.map(plain) });
    const seen = new Set(recent.map((item) => item.id));
    for (const { type, label, cap } of SECTIONS) {
        if (type === 'heading') continue;
        const own = entries
            .map((entry) => entry.item)
            .filter((item) => item.type === type && !seen.has(item.id))
            .slice(0, cap);
        if (own.length) sections.push({ type, label, results: own.map(plain) });
    }
    return sections;
}
