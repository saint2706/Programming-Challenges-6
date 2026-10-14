/**
 * The fuzzy matcher: does `pattern` appear as a subsequence of `text`, and how good is the best
 * alignment? An fzf-v2 / Sublime-style scorer, written out in full.
 *
 * Every matched character earns SCORE_MATCH plus a bonus that depends on where it sits: at the
 * start of a word, after a delimiter, at a camelCase hump, at the start of a number. Characters that
 * follow one another inherit the bonus of the first one in the run (typing "inst" for "Install"
 * scores like a word-start match four times, not once). Each gap between matched characters costs
 * GAP_START plus GAP_EXTEND per extra skipped character. The first pattern character counts double.
 *
 * The best alignment is found with a small dynamic program over (pattern char, text position), not
 * a greedy left-to-right scan, so "ar" in "bar/arch" matches the word-start "ar", not the "a" in
 * "bar" followed by the "r".
 *
 * Matching is case-insensitive and ignores diacritics ("cafe" finds "Café"). Positions are indices
 * into the text's code points (`Array.from(text)`), so astral characters never split.
 */

export const SCORE_MATCH = 16;
export const GAP_START = -3;
export const GAP_EXTEND = -1;
export const BONUS_BOUNDARY = 8;
export const BONUS_BOUNDARY_WHITE = 10;
export const BONUS_BOUNDARY_DELIMITER = 9;
export const BONUS_CAMEL = 7;
export const BONUS_CONSECUTIVE = -(GAP_START + GAP_EXTEND);
export const FIRST_CHAR_MULTIPLIER = 2;

/** Longest pattern and text the alignment runs on; see `fuzzyMatch` for what happens beyond. */
export const MAX_PATTERN = 32;
export const MAX_TEXT = 256;

const NEG = -1e9;

// Character classes: what comes before a character decides its bonus.
const WHITE = 0;
const NONWORD = 1;
const DELIMITER = 2;
const LOWER = 3;
const UPPER = 4;
const NUMBER = 5;

const DELIMITERS = new Set('/,:;|-_.'.split('').map((c) => c.charCodeAt(0)));
const RE_UPPER = /\p{Lu}/u;
const RE_LETTER = /[\p{L}\p{M}]/u;
const RE_NUMBER = /\p{N}/u;
const RE_SPACE = /\s/u;

function classOf(ch) {
    const code = ch.charCodeAt(0);
    if (code < 128) {
        if (code >= 97 && code <= 122) return LOWER;
        if (code >= 65 && code <= 90) return UPPER;
        if (code >= 48 && code <= 57) return NUMBER;
        if (code === 32 || (code >= 9 && code <= 13)) return WHITE;
        return DELIMITERS.has(code) ? DELIMITER : NONWORD;
    }
    if (RE_UPPER.test(ch)) return UPPER;
    if (RE_LETTER.test(ch)) return LOWER; // scripts without case count as lower case
    if (RE_NUMBER.test(ch)) return NUMBER;
    return RE_SPACE.test(ch) ? WHITE : NONWORD;
}

function bonusFor(prev, cls) {
    if (cls === LOWER || cls === UPPER || cls === NUMBER) {
        if (prev === WHITE) return BONUS_BOUNDARY_WHITE;
        if (prev === DELIMITER) return BONUS_BOUNDARY_DELIMITER;
        if (prev === NONWORD) return BONUS_BOUNDARY;
        if (prev === LOWER && cls === UPPER) return BONUS_CAMEL;
        if (prev !== NUMBER && cls === NUMBER) return BONUS_CAMEL;
        return 0;
    }
    if (cls === WHITE) return BONUS_BOUNDARY_WHITE;
    return BONUS_BOUNDARY; // matching a delimiter or symbol itself ("/", "-")
}

/** One character reduced for comparison: lower case, diacritics dropped, a single code point. */
function foldChar(ch) {
    const code = ch.charCodeAt(0);
    if (code < 128) return code >= 65 && code <= 90 ? code + 32 : code;
    const base = ch.normalize('NFD').toLowerCase();
    return base.codePointAt(0);
}

/**
 * A text ready for matching. Do this once per item (the index does) and reuse it for every query:
 * folding and bonus lookup are the per-character costs, and they do not depend on the query.
 */
export function prepareText(text) {
    const chars = Array.from(text);
    const n = chars.length;
    const codes = new Uint32Array(n);
    const bonus = new Uint8Array(n);
    let prev = WHITE;
    for (let i = 0; i < n; i += 1) {
        const cls = classOf(chars[i]);
        codes[i] = foldChar(chars[i]);
        bonus[i] = bonusFor(prev, cls);
        prev = cls;
    }
    return { text, chars, codes, bonus, n };
}

/** A pattern as folded code points. Returns `null` when it is empty. */
export function preparePattern(pattern) {
    const codes = Array.from(pattern, foldChar);
    return codes.length ? Uint32Array.from(codes) : null;
}

// Scratch matrices, reused between calls (the matcher is synchronous, so nothing interleaves).
const score = new Int32Array(MAX_PATTERN * MAX_TEXT);
const chunk = new Uint8Array(MAX_PATTERN * MAX_TEXT); // bonus of the first char of the current run
const prevAt = new Int16Array(MAX_PATTERN * MAX_TEXT); // which text position the previous pattern char sat at

function isSubsequence(pat, hay) {
    let i = 0;
    for (let j = 0; j < hay.n && i < pat.length; j += 1) if (hay.codes[j] === pat[i]) i += 1;
    return i === pat.length;
}

/** Fallback for texts longer than MAX_TEXT: first-occurrence greedy scan, scored the same way. */
function greedy(pat, hay) {
    const positions = [];
    let j = 0;
    let total = 0;
    let run = 0;
    for (let i = 0; i < pat.length; i += 1) {
        while (j < hay.n && hay.codes[j] !== pat[i]) j += 1;
        const b = hay.bonus[j];
        const consecutive = positions.length > 0 && positions[positions.length - 1] === j - 1;
        run = consecutive ? Math.max(run, b, BONUS_CONSECUTIVE) : b;
        total += SCORE_MATCH + (i === 0 ? b * FIRST_CHAR_MULTIPLIER : run);
        if (!consecutive && positions.length > 0) total += GAP_START + GAP_EXTEND * (j - positions[positions.length - 1] - 2);
        positions.push(j);
        j += 1;
    }
    return { score: total, positions };
}

/**
 * Best alignment of a prepared pattern against a prepared text, or `null` when the pattern is not a
 * subsequence. `positions` are the matched code-point indices, ascending.
 */
export function matchPrepared(pat, hay) {
    const m = pat.length;
    const n = hay.n;
    if (m === 0 || m > n || m > MAX_PATTERN || !isSubsequence(pat, hay)) return null;
    if (n > MAX_TEXT) return greedy(pat, hay);

    const { codes, bonus } = hay;
    for (let i = 0; i < m; i += 1) {
        const row = i * n;
        let gap = NEG; // best score of pat[0..i-1] ending in a gap of 1+ chars just before j
        let gapFrom = -1; // where that alignment's last matched char sat
        for (let j = 0; j < n; j += 1) {
            if (i > 0 && j >= 2) {
                const open = score[row - n + j - 2] + GAP_START;
                const extend = gap + GAP_EXTEND;
                if (open >= extend) {
                    gap = open;
                    gapFrom = j - 2;
                } else gap = extend;
            }
            if (codes[j] !== pat[i]) {
                score[row + j] = NEG;
                continue;
            }
            const b = bonus[j];
            if (i === 0) {
                score[row + j] = SCORE_MATCH + b * FIRST_CHAR_MULTIPLIER;
                chunk[row + j] = b;
                prevAt[row + j] = -1;
                continue;
            }
            // Option A: pat[i] follows pat[i-1] directly, extending a run.
            let best = NEG;
            let first = b;
            let from = -1;
            const diag = j >= 1 ? score[row - n + j - 1] : NEG;
            if (diag > NEG / 2) {
                first = chunk[row - n + j - 1];
                if (b >= BONUS_BOUNDARY && b > first) first = b;
                best = diag + SCORE_MATCH + Math.max(b, BONUS_CONSECUTIVE, first);
                from = j - 1;
            }
            // Option B: pat[i] comes after a gap.
            if (gap > NEG / 2) {
                const viaGap = gap + SCORE_MATCH + b;
                if (viaGap > best) {
                    best = viaGap;
                    first = b;
                    from = gapFrom;
                }
            }
            score[row + j] = best > NEG / 2 ? best : NEG;
            chunk[row + j] = first;
            prevAt[row + j] = from;
        }
    }

    const last = (m - 1) * n;
    let bestScore = NEG;
    let at = -1;
    for (let j = 0; j < n; j += 1) {
        if (score[last + j] > bestScore) {
            bestScore = score[last + j];
            at = j;
        }
    }
    if (at < 0) return null;
    const positions = new Array(m);
    for (let i = m - 1; i >= 0; i -= 1) {
        positions[i] = at;
        at = prevAt[i * n + at];
    }
    return { score: bestScore, positions };
}

/**
 * `fuzzyMatch('inst', 'Install the script')` -> `{ score, positions }`, or `null` for no match.
 * Convenience over `matchPrepared` for one-off use and tests; the search path prepares once.
 */
export function fuzzyMatch(pattern, text) {
    const pat = preparePattern(pattern);
    return pat ? matchPrepared(pat, prepareText(text)) : null;
}

/**
 * Split `text` into alternating runs for rendering: `[{ text, match }]`. Built from code points, so
 * the caller can append each run with `textContent` and never needs `innerHTML`.
 */
export function highlightRuns(text, positions) {
    const chars = Array.from(text);
    const marked = new Set(positions);
    const runs = [];
    for (let i = 0; i < chars.length; i += 1) {
        const match = marked.has(i);
        const tail = runs[runs.length - 1];
        if (tail && tail.match === match) tail.text += chars[i];
        else runs.push({ text: chars[i], match });
    }
    return runs;
}
