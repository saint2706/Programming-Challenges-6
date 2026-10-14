// How often does the matcher's alignment hit the best possible score? Compares it with a brute-force
// enumeration (tests/oracle.js) on 40,000 seeded random small cases. Run: node bench/optimality.mjs
import { fuzzyMatch } from '../src/fuzzy.js';
import { allAlignments } from '../tests/oracle.js';

let seed = 1234567;
const random = () => (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2 ** 32;
const TEXT_CHARS = 'abAB1 -/_.';
const PATTERN_CHARS = 'abAB1';
const draw = (chars, length) => Array.from({ length }, () => chars[Math.floor(random() * chars.length)]).join('');

let matched = 0;
let optimal = 0;
let worstGap = 0;
for (let i = 0; i < 40000; i += 1) {
    const text = draw(TEXT_CHARS, 3 + Math.floor(random() * 10));
    const pattern = draw(PATTERN_CHARS, 1 + Math.floor(random() * 4));
    const match = fuzzyMatch(pattern, text);
    if (!match) continue;
    const best = Math.max(...allAlignments(pattern, text).map((a) => a.score));
    matched += 1;
    if (match.score === best) optimal += 1;
    else worstGap = Math.max(worstGap, best - match.score);
}
console.log(`${optimal} of ${matched} matching cases hit the best score (${((optimal / matched) * 100).toFixed(3)}%); worst shortfall ${worstGap} point(s)`);
