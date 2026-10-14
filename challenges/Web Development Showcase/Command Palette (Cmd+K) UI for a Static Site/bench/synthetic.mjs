/**
 * A large synthetic index (documentation-style titles, identifiers and paths) and queries derived
 * from known targets. Seeded, so every run and every matcher sees exactly the same data.
 *
 * Word use is skewed (a few words are everywhere, many are rare), like real documentation, and a
 * single-word query is cut from the rarest word of its target. Even so, a query such as "conf" fits
 * many titles, so no matcher can put the one target first every time. That hurts all of them alike:
 * the numbers compare matchers, they are not an absolute accuracy.
 */

const WORDS = `
access account action activate adapter address admin agent alert alias allocate analytics anchor
api archive argument array asset async audit auth backup balance banner batch binary binding block
bootstrap branch breakpoint browser bucket buffer build bundle cache calendar callback cancel
canvas capture cascade catalog certificate channel checkout chunk circuit claim classify cleanup
client clipboard cluster codec collection column command commit compile component compress config
connect console constant container context contract convert cookie coordinate counter coverage
credential cursor custom dashboard database dataset deadline debug decode default delegate delete
deploy deprecate descriptor device diagnose dialog digest directory disable dispatch document domain
download draft driver duplicate dynamic element email embed enable encode encrypt endpoint engine
entity environment error escape event exception execute expire export extension factory failure
feature fetch field filter fingerprint firewall flag flatten folder font footer format forward
fragment frame gateway generate geometry gesture global gradient grant graph group guard handler
hash header health heartbeat history hook host hydrate icon identity image import index inherit
inject input inspect install instance integer interface interval invalid invoice isolate iterator
job journal kernel keyboard label language latency layout lazy ledger library lifecycle limit
listener locale lock logger login manifest mapping margin markdown matrix media memory merge
message metadata method metric migrate mirror module monitor mutation namespace navigate network
node notification object offset onboarding operator option origin output override package padding
pagination parameter parser partition password patch payload permission pipeline placeholder
plugin pointer policy polling pool portal preference prefetch preview primary priority profile
progress promise property protocol provider proxy publish query queue quota radio range rate reader
realtime record redirect reducer refresh region registry release remote render replica repository
request reset resolve resource response restore retry revision role rollback route runtime sandbox
scale schedule schema scope script search secret section selector sequence serialize server
session setting shadow shard sidebar signal signature snapshot socket sort source spacing
specification split stack state static storage stream string style subscribe summary support switch
symbol sync syntax table tag target template tenant terminal theme thread threshold throttle
timeline timeout token toolbar tooltip trace transaction transform transition tree trigger tunnel
upgrade upload usage validate variable vector version viewport virtual visibility webhook widget
window worker workflow workspace wrapper
`
    .split(/\s+/u)
    .filter(Boolean);

const VERBS = ['Configure', 'Create', 'Debug', 'Deploy', 'Enable', 'Handle', 'Manage', 'Migrate', 'Monitor', 'Optimize', 'Reset', 'Set up', 'Understand', 'Use', 'Write'];
const SECTIONS = ['Guides', 'Reference', 'Tutorials', 'Concepts', 'Changelog', 'API', 'Cookbook', 'Admin'];

/** mulberry32: a tiny seeded generator. */
export function rng(seed) {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

const cap = (word) => word[0].toUpperCase() + word.slice(1);

export function makeIndex(count, seed = 7) {
    const rand = rng(seed);
    const pick = (list) => list[Math.floor(rand() * list.length)];
    const pickWord = () => WORDS[Math.floor(WORDS.length * rand() ** 2.5)]; // skewed towards the head of the list
    const seen = new Set();
    const items = [];
    while (items.length < count) {
        const kind = rand();
        const [a, b, c] = [pickWord(), pickWord(), pickWord()];
        let title;
        if (kind < 0.45) title = `${pick(VERBS)} ${a} ${b}`;
        else if (kind < 0.65) title = `${cap(a)} ${b} and ${c}`;
        else if (kind < 0.8) title = `use${cap(a)}${cap(b)}`; // a hook-style identifier
        else if (kind < 0.9) title = `api/v${1 + Math.floor(rand() * 3)}/${a}s/{id}/${b}`; // a path
        else title = `${cap(a)} ${b} ${1 + Math.floor(rand() * 20)}.${Math.floor(rand() * 10)}`;
        if (seen.has(title)) continue;
        seen.add(title);
        items.push({
            id: `s:${items.length}`,
            type: rand() < 0.3 ? 'page' : 'heading',
            title,
            hint: pick(SECTIONS),
            ...(rand() < 0.25 && { keywords: [pick(WORDS), pick(WORDS)] }),
            url: `/docs/${items.length}`,
        });
    }
    return items;
}

const words = (title) => title.split(/[\s/{}_.-]+|(?<=[a-z])(?=[A-Z])/u).filter(Boolean);
const swap = (word) => (word.length < 4 ? word : `${word.slice(0, 1)}${word[2]}${word[1]}${word.slice(3)}`);

/** Queries a visitor might type to reach `item`, by kind. Returns `null` when the title is too short for it. */
function derive(kind, title, rand, uses) {
    const parts = words(title).filter((w) => /^[A-Za-z]{4,}$/u.test(w));
    if (parts.length === 0) return null;
    const word = parts.map((w) => w.toLowerCase()).sort((x, y) => (uses.get(x) ?? 0) - (uses.get(y) ?? 0))[0];
    switch (kind) {
        case 'prefix':
            return word.slice(0, Math.max(4, Math.ceil(word.length * 0.6)));
        case 'initials':
            return parts.length >= 3 ? parts.map((w) => w[0]).join('').toLowerCase() : null;
        case 'infix':
            return word.length >= 6 ? word.slice(2, 2 + Math.max(4, word.length - 4)) : null;
        case 'skipped':
            return word.length >= 6 ? word.replace(/[aeiou]/gu, (m, i) => (i === 0 ? m : '')) : null;
        case 'multiword':
            return parts.length >= 2 ? `${parts[0].slice(0, 4)} ${parts[parts.length - 1].slice(0, 4)}`.toLowerCase() : null;
        case 'typo':
            return word.length >= 6 ? swap(word) : null;
        default:
            return null;
    }
}

export const KINDS = ['prefix', 'initials', 'infix', 'skipped', 'multiword', 'typo'];

/** About `perKind` queries of each kind, each with the id of the item it was derived from. */
export function makeQueries(items, perKind = 60, seed = 11) {
    const rand = rng(seed);
    const uses = new Map();
    for (const { title } of items) for (const w of words(title)) uses.set(w.toLowerCase(), (uses.get(w.toLowerCase()) ?? 0) + 1);
    const queries = [];
    for (const kind of KINDS) {
        let made = 0;
        let guard = 0;
        while (made < perKind && guard < perKind * 200) {
            guard += 1;
            const item = items[Math.floor(rand() * items.length)];
            const q = derive(kind, item.title, rand, uses);
            if (!q || q.length < 2) continue;
            queries.push({ kind, q, want: [item.id] });
            made += 1;
        }
    }
    return queries;
}
