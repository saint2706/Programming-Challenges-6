/**
 * Hand-labelled queries for the real demo-site index (the 38 pages and headings plus the 5 actions).
 * Written before the matchers were compared, one kind at a time, as a visitor might type them.
 * `want` lists every id that is an acceptable answer; the score is the rank of the best of them.
 *
 * Some queries are ones the scorer is expected to lose (typos that are not a subsequence, words
 * that appear nowhere in the item): they stay in, because dropping them would flatter it.
 */
export const REAL_QUERIES = [
    // prefix of a word in the title
    { kind: 'prefix', q: 'keyb', want: ['p:keyboard.html'] },
    { kind: 'prefix', q: 'access', want: ['p:accessibility.html'] },
    { kind: 'prefix', q: 'getting', want: ['p:getting-started.html'] },
    { kind: 'prefix', q: 'reduced', want: ['h:accessibility.html#reduced-motion'] },
    { kind: 'prefix', q: 'diacr', want: ['h:matching.html#diacritics'] },
    { kind: 'prefix', q: 'subseq', want: ['h:matching.html#subsequence'] },
    // initials of the words
    { kind: 'initials', q: 'gs', want: ['p:getting-started.html'] },
    { kind: 'initials', q: 'rm', want: ['h:accessibility.html#reduced-motion'] },
    { kind: 'initials', q: 'mw', want: ['h:matching.html#multiple-words'] },
    { kind: 'initials', q: 'pn', want: ['h:keyboard.html#platform-notes'] },
    { kind: 'initials', q: 'gti', want: ['h:getting-started.html#generate-index'] },
    { kind: 'initials', q: 'wut', want: ['h:getting-started.html#wire-trigger'] },
    { kind: 'initials', q: 'ccd', want: ['h:matching.html#camel-case'] },
    // letters from inside a word
    { kind: 'infix', q: 'ime', want: ['h:accessibility.html#ime'] },
    { kind: 'infix', q: 'sequence', want: ['h:matching.html#subsequence'] },
    { kind: 'infix', q: 'board', want: ['p:keyboard.html'] },
    { kind: 'infix', q: 'listbox', want: ['h:accessibility.html#combobox-pattern'] },
    { kind: 'infix', q: 'stall', want: ['h:getting-started.html#install'] },
    // some letters skipped
    { kind: 'skipped', q: 'accsblty', want: ['p:accessibility.html'] },
    { kind: 'skipped', q: 'kybrd', want: ['p:keyboard.html'] },
    { kind: 'skipped', q: 'shrtcts', want: ['p:keyboard.html'] },
    { kind: 'skipped', q: 'cmbbx', want: ['h:accessibility.html#combobox-pattern'] },
    { kind: 'skipped', q: 'annncmnts', want: ['h:accessibility.html#announcements'] },
    // more than one word
    { kind: 'multiword', q: 'match gaps', want: ['h:matching.html#gaps'] },
    { kind: 'multiword', q: 'recent stored', want: ['h:faq.html#recent-items'] },
    { kind: 'multiword', q: 'reduced motion', want: ['h:accessibility.html#reduced-motion'] },
    { kind: 'multiword', q: 'install getting', want: ['h:getting-started.html#install'] },
    { kind: 'multiword', q: 'touch scr', want: ['h:accessibility.html#touch'] },
    { kind: 'multiword', q: 'wire trig', want: ['h:getting-started.html#wire-trigger'] },
    // a word that is only in the keywords, not the title
    { kind: 'keyword', q: 'hotkeys', want: ['p:keyboard.html'] },
    { kind: 'keyword', q: 'a11y', want: ['p:accessibility.html'] },
    { kind: 'keyword', q: 'fzf', want: ['p:matching.html'] },
    { kind: 'keyword', q: 'tutorial', want: ['p:getting-started.html'] },
    { kind: 'keyword', q: 'dark', want: ['a:theme'] },
    { kind: 'keyword', q: 'clipboard', want: ['a:copy-link'] },
    // typos
    { kind: 'typo', q: 'acessibility', want: ['p:accessibility.html'] }, // a letter missing
    { kind: 'typo', q: 'matchng', want: ['p:matching.html', 'h:matching.html#subsequence'] }, // a letter missing
    { kind: 'typo', q: 'keybaord', want: ['p:keyboard.html'] }, // letters swapped
    { kind: 'typo', q: 'shortcust', want: ['p:keyboard.html'] }, // letters swapped
    { kind: 'typo', q: 'gettnig', want: ['p:getting-started.html'] }, // letters swapped
    { kind: 'typo', q: 'accesibility', want: ['p:accessibility.html'] }, // a letter missing
    { kind: 'typo', q: 'bonnuses', want: ['h:matching.html#bonuses'] }, // a letter doubled
    { kind: 'typo', q: 'dacritics', want: ['h:matching.html#diacritics'] }, // a letter missing
];
