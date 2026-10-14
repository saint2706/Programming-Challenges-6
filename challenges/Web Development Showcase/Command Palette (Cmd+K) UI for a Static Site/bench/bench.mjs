// Ranking quality and latency of the palette's matcher against Fuse.js and uFuzzy, on the real demo
// index (hand-labelled queries) and on synthetic indexes of 1,000 and 20,000 entries (derived
// queries). Run with `npm run bench`; the tables go to stdout.
import { readFileSync } from 'node:fs';
import { cpus } from 'node:os';
import { fileURLToPath } from 'node:url';
import uFuzzy from '@leeoniya/ufuzzy';
import Fuse from 'fuse.js';
import { ACTIONS } from '../src/actions.js';
import { createSearch } from '../src/search.js';
import { REAL_QUERIES } from './queries.mjs';
import { makeIndex, makeQueries } from './synthetic.mjs';

const LIMIT = 50; // what a palette would ever show or scan
const PASSES = 3;
const SHORT = ['e', 's', 'a', 'o', 'co', 'st', 're', 'in', 'ap', 'se', 'tr', 'ma']; // one and two letters: matches nearly everything

const here = (path) => fileURLToPath(new URL(path, import.meta.url));
const real = JSON.parse(readFileSync(here('../src/generated/search-index.json'), 'utf8')).items;
const realItems = [...ACTIONS.map(({ run, ...item }) => item), ...real];
const keywordText = (item) => (item.keywords ?? []).join(' ');

/** Each matcher: build(items) -> (query) -> ranked ids. */
const MATCHERS = [
    {
        name: 'this palette',
        build(items) {
            const search = createSearch(items);
            return (q) => search.rank(q, LIMIT).map((hit) => hit.item.id);
        },
    },
    {
        name: 'Fuse.js (defaults)',
        build(items) {
            const docs = items.map((item) => ({ id: item.id, title: item.title, hint: item.hint ?? '', kw: keywordText(item) }));
            const fuse = new Fuse(docs, { keys: ['title', 'hint', 'kw'], includeScore: true });
            return (q) => fuse.search(q, { limit: LIMIT }).map((r) => r.item.id);
        },
    },
    {
        // The best of 20 option sets tried on the demo queries (threshold 0.3-0.7, ignoreLocation, extended
        // search, field weights), so it is as generous to Fuse as the demo set allows.
        name: 'Fuse.js (best of 20)',
        build(items) {
            const docs = items.map((item) => ({ id: item.id, title: item.title, hint: item.hint ?? '', kw: keywordText(item) }));
            const fuse = new Fuse(docs, {
                keys: [{ name: 'title', weight: 3 }, { name: 'kw', weight: 2 }, { name: 'hint', weight: 1 }],
                ignoreLocation: true,
                threshold: 0.7,
                includeScore: true,
            });
            return (q) => fuse.search(q, { limit: LIMIT }).map((r) => r.item.id);
        },
    },
    ...[
        ['uFuzzy (default)', { intraMode: 0 }],
        ['uFuzzy (typo mode)', { intraMode: 1 }],
    ].map(([name, options]) => ({
        name,
        build(items) {
            const haystack = items.map((item) => [item.title, item.hint ?? '', keywordText(item)].join(' ').trim());
            const uf = new uFuzzy(options);
            return (q) => {
                // infoThresh is raised so uFuzzy ranks every match; at its default 1,000 it stops ranking.
                const [idxs, info, order] = uf.search(haystack, q, 3, 1e9);
                if (!idxs) return [];
                const ranked = info && order ? order.map((o) => info.idx[o]) : idxs;
                return ranked.slice(0, LIMIT).map((i) => items[i].id);
            };
        },
    })),
];

const percentile = (sorted, p) => sorted[Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1)];
const ms = (n) => (n < 0.1 ? `${(n * 1000).toFixed(0)} µs` : `${n.toFixed(2)} ms`);
const pct = (n) => `${(n * 100).toFixed(0)}%`;

const kindsOf = (queries) => [...new Set(queries.map((x) => x.kind))];

function quality(run, queries) {
    const rows = queries.map(({ kind, q, want }) => {
        const ids = run(q);
        const at = ids.findIndex((id) => want.includes(id));
        return { kind, rank: at < 0 ? Infinity : at + 1 };
    });
    const stat = (list) => ({
        n: list.length,
        top1: list.filter((r) => r.rank <= 1).length / list.length,
        top3: list.filter((r) => r.rank <= 3).length / list.length,
        top10: list.filter((r) => r.rank <= 10).length / list.length,
        mrr: list.reduce((sum, r) => sum + (r.rank === Infinity ? 0 : 1 / r.rank), 0) / list.length,
        miss: list.filter((r) => r.rank === Infinity).length / list.length,
    });
    return { all: stat(rows), byKind: Object.fromEntries(kindsOf(queries).map((kind) => [kind, stat(rows.filter((r) => r.kind === kind))])) };
}

function latency(run, queries, passes = PASSES) {
    for (const q of queries) run(q); // warm up
    const samples = [];
    for (let pass = 0; pass < passes; pass += 1) {
        for (const q of queries) {
            const t0 = performance.now();
            run(q);
            samples.push(performance.now() - t0);
        }
    }
    samples.sort((a, b) => a - b);
    return { p50: percentile(samples, 50), p95: percentile(samples, 95), max: samples.at(-1) };
}

function buildTime(matcher, items) {
    const times = [];
    for (let i = 0; i < 5; i += 1) {
        const t0 = performance.now();
        matcher.build(items);
        times.push(performance.now() - t0);
    }
    return times.sort((a, b) => a - b)[2];
}

const results = {
    env: {
        node: process.version,
        cpu: cpus()[0].model.trim(),
        fuse: JSON.parse(readFileSync(here('../node_modules/fuse.js/package.json'), 'utf8')).version,
        ufuzzy: JSON.parse(readFileSync(here('../node_modules/@leeoniya/ufuzzy/package.json'), 'utf8')).version,
    },
    datasets: [],
};

// At 20,000 entries each word is in ~150 titles, so a one-word query fits far more titles than any
// matcher could rank the intended one into the top 50: latency is measured there, quality is not.
const synthetic = (count, scored) => {
    const items = makeIndex(count);
    return { label: `synthetic ${count.toLocaleString('en-US')}`, items, queries: makeQueries(items), scored };
};
const DATASETS = [{ label: 'demo site', items: realItems, queries: REAL_QUERIES, scored: true }, synthetic(1000, true), synthetic(20000, false)];

for (const { label, items, queries, scored } of DATASETS) {
    const entry = { label, entries: items.length, queries: queries.length, matchers: {} };
    console.log(`\n=== ${label}: ${items.length} entries, ${queries.length} labelled queries ===`);
    console.log('matcher                top1  top3  top10  MRR   miss   build      p50        p95        short p95');
    for (const matcher of MATCHERS) {
        const run = matcher.build(items);
        const q = scored ? quality(run, queries) : null;
        const lat = latency(run, queries.map((x) => x.q));
        const short = latency(run, SHORT, 8);
        const build = buildTime(matcher, items);
        entry.matchers[matcher.name] = { quality: q, latency: lat, shortLatency: short, buildMs: build };
        const quals = q
            ? `${pct(q.all.top1).padStart(4)}  ${pct(q.all.top3).padStart(4)}  ${pct(q.all.top10).padStart(5)}  ${q.all.mrr.toFixed(2)}  ${pct(q.all.miss).padStart(4)}`
            : '   -     -      -     -      -';
        console.log(
            `${matcher.name.padEnd(22)} ${quals}   ${ms(build).padEnd(9)}  ${ms(lat.p50).padEnd(9)}  ${ms(lat.p95).padEnd(9)}  ${ms(short.p95)}`,
        );
    }
    if (!scored) {
        results.datasets.push(entry);
        continue;
    }
    console.log('\ntop-3 hit rate by kind of query');
    const kinds = kindsOf(queries);
    console.log(`matcher                ${kinds.map((k) => k.padEnd(10)).join(' ')}`);
    for (const matcher of MATCHERS) {
        const { byKind } = entry.matchers[matcher.name].quality;
        console.log(`${matcher.name.padEnd(22)} ${kinds.map((k) => `${pct(byKind[k].top3)} (${byKind[k].n})`.padEnd(10)).join(' ')}`);
    }
    results.datasets.push(entry);
}

console.log(`\n${results.env.node} on ${results.env.cpu}; fuse.js ${results.env.fuse}, uFuzzy ${results.env.ufuzzy}`);
