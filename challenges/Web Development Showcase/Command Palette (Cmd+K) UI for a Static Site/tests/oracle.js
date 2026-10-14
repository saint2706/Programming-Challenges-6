import {
    BONUS_BOUNDARY,
    BONUS_CONSECUTIVE,
    FIRST_CHAR_MULTIPLIER,
    GAP_EXTEND,
    GAP_START,
    SCORE_MATCH,
    preparePattern,
    prepareText,
} from '../src/fuzzy.js';

/** The scoring rules written out directly, for one given alignment (the matcher's DP must agree). */
export function scoreAlignment(positions, hay) {
    let total = 0;
    let first = 0;
    let previous = -2;
    positions.forEach((j, i) => {
        const b = hay.bonus[j];
        let used = b;
        if (i > 0 && j === previous + 1) {
            if (b >= BONUS_BOUNDARY && b > first) first = b;
            used = Math.max(b, BONUS_CONSECUTIVE, first);
        } else {
            first = b;
            if (i > 0) total += GAP_START + GAP_EXTEND * (j - previous - 2);
        }
        total += SCORE_MATCH + (i === 0 ? b * FIRST_CHAR_MULTIPLIER : used);
        previous = j;
    });
    return total;
}

/** Every alignment of `pattern` in `text`, by brute force: `{ positions, score }[]`. */
export function allAlignments(pattern, text) {
    const pat = preparePattern(pattern);
    const hay = prepareText(text);
    const out = [];
    const walk = (i, from, chosen) => {
        if (i === pat.length) {
            out.push({ positions: chosen, score: scoreAlignment(chosen, hay) });
            return;
        }
        for (let j = from; j < hay.n; j += 1) if (hay.codes[j] === pat[i]) walk(i + 1, j + 1, [...chosen, j]);
    };
    walk(0, 0, []);
    return out;
}
