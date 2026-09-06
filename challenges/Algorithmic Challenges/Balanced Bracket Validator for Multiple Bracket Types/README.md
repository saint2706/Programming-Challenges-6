# Balanced Bracket Validator for Multiple Bracket Types

**Category:** Algorithmic Challenges
**Difficulty:** B (brief: "stack-based, support custom/nested pair definitions")

**Status:** Implemented (Python)

The interview version of this problem is one stack, three hard-coded pairs, ten
lines. That version is wrong about almost every real file you point it at:

```c
char *s = "))))";      /* the naive validator reports four stray closers */
/* ( */                /* ...and one unclosed paren */
```

So the stack stays — it is the right data structure and nothing here replaces
it — and everything *around* it gets generalized. The result is a validator
you could actually put behind an editor's rainbow-brackets feature or a
pre-commit hook.

## What the generalization has to cover

| Problem                    | What breaks without it                                     | How it's handled                                                         |
| -------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------ |
| Multi-character delimiters | `<` shadows `<!--`, so every HTML comment looks unbalanced | Leftmost-**longest** matching                                            |
| Strings and comments       | A `)` in a string literal is text, not a delimiter         | `opaque=True` suppresses the whole delimiter set until the region closes |
| Escapes                    | `"a\"b"` is one string; naive scanning ends it early       | Per-pair `escape`, consumed with the character it protects               |
| Self-pairing delimiters    | `"`, `$`, ` ``` ` open *and* close with the same lexeme    | Role resolved against the stack, not the lexeme                          |
| Non-nesting pairs          | `/* /* */` is one comment in C, not two                    | `nestable=False`                                                         |
| Grammar shape              | `{[}]` is "balanced" to a naive stack of one type          | `may_contain` whitelists direct children                                 |
| Line comments              | `//` ends at a newline — or at EOF                         | `optional_close=True`                                                    |
| Huge inputs                | A 10 GB file shouldn't need 10 GB of RAM                   | `Validator.feed()` streams; memory is O(depth)                           |
| Bad input                  | Stopping at the first error makes a linter useless         | Recovery heuristic, every fault reported                                 |

## The one design decision worth arguing about

**Scanning.** The obvious "optimal" answer is Aho–Corasick: O(n + z) over all
delimiters, textbook-correct. I didn't use it, and the reason is worth stating
plainly rather than hiding.

Aho–Corasick reports every match *ending* at each position. This validator
needs the longest match *starting* at the current position, and it needs to
suspend the whole automaton the instant it enters an opaque region — you can't
usefully keep matching `{` and `(` while inside a string literal. Both of those
fight the automaton's shape, and the fix (reverse indexing, or re-seeding the
automaton at every region boundary) costs more than it saves.

What's used instead: one compiled alternation over every lexeme, **ordered
longest-first**. Python's `re` takes the leftmost match and, at that position,
the first alternative that matches — so the ordering buys exact
leftmost-longest semantics, and `search()` skips inert text in C rather than in
a Python loop over characters. Inside an opaque region a second, two-alternative
pattern (`close|escape`) jumps straight to whatever can end it.

Asymptotically this is O(n · L) for maximum lexeme length L, which is a small
constant per spec. Practically it is the fastest thing available in Python and
about ten lines of code. That trade is the honest answer here; on a compiled
target with hundreds of delimiters, Aho–Corasick would win.

## Error recovery

Reporting only the first fault makes a linter unusable on a file that has two.
On a closer that the stack top doesn't want, the validator does what production
parsers do:

- **The closer matches a frame further down the stack** → everything above it
  was left open. Report each of those as `unclosed`, resynchronize at that
  depth. `{ [ }` blames the `[`, not the `}` — same as clang and rustc.
- **The closer matches nothing on the stack** → the closer is the stray token.
  Report it and *discard it*, leaving the stack intact so the remaining
  diagnostics stay meaningful instead of cascading.

Diagnostics carry an absolute offset, 1-based line/column, the lexeme, the pair
name, and a cross-reference to the related position (which opener, or which
closer arrived first):

```
$ uv run python brackets.py --spec c broken.c
broken.c: FAILED (max depth 3)
  2:10: unclosed: '[' is never closed; '}' closed 'curly' first (opened at 4:1)
        int a[2 = {1};
             ^
```

`--json` emits the same thing machine-readably.

## What an adversarial pass turned up

Four bugs that only a deliberate hunt for edge cases would find, all now fixed
and pinned by tests:

**`ok` could lie.** `max_diagnostics` caps the reported list, and `ok` was
derived from that list — so `validate(text, max_diagnostics=0)` called *any*
input valid, including `"(((("`. Faults are now counted separately from the
diagnostics that get listed, and `ok` follows the count. `Report` also carries
`fault_count` and `truncated`, so a capped report says how much it hid.

**The depth limit cascaded.** Exceeding `max_depth` dropped the frame and
carried on, which turned every subsequent closer into a spurious
"unexpected close": a limit of 3 on 20 nested parens produced 1 real
diagnostic and 17 pieces of noise. The limit exists to bound work, so hitting
it is now fatal — one diagnostic, scan abandoned.

**Two Pair configurations were silently unusable.** `escape == close` makes the
scanner read every closer as an escape, so the region can never end — an
unterminated string swallowing the rest of the file, reported as one confusing
fault at EOF. An `escape` on a non-opaque pair is simply ignored. Both are now
rejected at construction with a message that says why.

**Diagnostics came back out of order.** Unclosed frames are discovered at end
of input but belong where their opener is. They are now sorted by offset, which
is what makes the caret output readable top to bottom.

Also probed and found already correct, now pinned by tests: a lexeme shared as
one pair's opener and another's closer, an opener that is a prefix of its own
closer, CRLF line endings, `finish()` called twice, and a 500 KB single line.

## A subtlety that took a bug to find

An opaque pair's *closer* is deliberately not registered as a globally visible
delimiter. It is only recognized by that pair's own scanner, from inside the
region.

Without that rule, Python's `#`…`\n` comment pair makes **every newline in the
file** a stray closer. The same rule means a `*/` outside a comment isn't
flagged — which is correct for this tool: that's a lexer error, not a nesting
error, and inventing a bracket fault for it would be worse than silence.

## Streaming

`Validator.feed()` accepts any chunking and reports absolute positions over the
concatenation. The only state that must survive a chunk boundary is the
delimiter stack plus a hold-back buffer of `max(len(lexeme), len(escape) + 1)`
characters — the shortest window in which a token could still be ambiguous.

The escape case is the sharp edge: a chunk that ends in the middle of `\"`
must resume *before* the backslash, never between it and the quote, or the
scanner re-reads the quote as a string terminator and desynchronizes the entire
rest of the file. The tests check streaming against the whole-string pass over
six specs at eight chunk sizes down to one character, which is how that bug got
caught.

## Editor-flavored extras

The same single pass supports three features that usually get written
separately:

```python
matching_index("x <!-- y --> z", 3, "html")   # 9  -- jump-to-match, from *inside* "<!--"
longest_balanced_span("())((()))")            # (3, 9)  -- "longest valid parentheses", any pair type
auto_close('return g("a", [1, {2:', "python") # '}])'  -- what an editor would insert
```

`auto_close` works on a *partial* buffer precisely because the validator never
needs to see end-of-input to know what is open.

## Presets and custom grammars

Built in: `plain`, `angle`, `c`, `python`, `json`, `html`, `latex`, `markdown`,
`unicode` (CJK corner brackets, fullwidth forms, guillemets, math angles).

The LaTeX preset is the best showcase — it exercises multi-character openers
(`\begin`/`\end`), self-pairing delimiters (`$`, `$$`), non-nesting rules, and
`may_contain` (math mode may hold braces and brackets, but not more math mode).

Custom grammars are data, not code:

```json
{"name": "pascal", "pairs": [
  {"open": "begin", "close": "end", "name": "block"},
  {"open": "{", "close": "}", "name": "comment", "opaque": true}
]}
```

```bash
uv run python brackets.py --spec-file pascal.json src.pas
uv run python brackets.py --spec c --dump-spec        # a preset, as editable JSON
```

## Performance

Measured on this repository's own source, validated against the `python` spec:

| Mode                                  | Throughput |
| ------------------------------------- | ---------- |
| `validate()` (collects matched spans) | ~5–6 MB/s  |
| `validate_stream()` (spans off)       | ~14 MB/s   |

The gap is span construction, not scanning — profiling puts ~90% of the time in
per-token Python work and only ~7% in `re.search`, which is exactly the shape
you want: the C engine skips the boring 99% of the file, and Python only wakes
up for the delimiters. 200k-deep nesting is handled iteratively, so there is no
recursion limit to hit.

## Where this is actually used

This is the challenge with the shortest distance to production code, because
almost every tool that touches source text needs exactly this — and needs the
generalized version, not the interview version.

**Editors.** Bracket matching, jump-to-match, rainbow brackets, auto-close and
code folding all run a delimiter scanner over the buffer on every keystroke.
All of them need opaque-region handling: a `)` inside a string literal must not
highlight, and one unterminated `"` must not make the rest of the file look
unbalanced.

**Template languages.** Jinja's `{% %}` and `{{ }}`, Handlebars, ERB's `<% %>`,
Go templates and Liquid are all multi-character delimiters that shadow
single-character ones. That is exactly the leftmost-longest matching problem,
and exactly what breaks a naive validator on `<!--` versus `<`.

**Cheap validation before expensive parsing.** Checking that a 10 GB JSON dump
or a database export was not truncated does not require parsing it. A streaming
delimiter check answers "did this file get cut off" in one pass with O(depth)
memory, which is what `Validator.feed()` is for. The same check in a pre-commit
hook catches an unbalanced brace before CI spends ten minutes rediscovering it.

**Error recovery is what makes a linter usable.** Blaming the `[` in `{ [ }`
rather than the `}`, and resynchronizing instead of giving up, is what clang and
rustc do — and it is the only reason a compiler reports twenty errors per run
instead of one. A validator that stops at the first fault turns every build into
a serial bisect.

**Security-adjacent.** Unbalanced or truncated delimiters are a signal in
template-injection detection and in spotting payloads that close a quoted
context early. The escape handling that makes `"a\"b"` parse as one string is
the same machinery that stops a scanner being fooled by one.

Tree-sitter and TextMate grammars solve a superset of this problem. The value of
the standalone version is that it needs no grammar for the language — only a
delimiter spec — so it works on the config format nobody has written a parser
for yet.

## Run it

```bash
cd "challenges/Algorithmic Challenges/Balanced Bracket Validator for Multiple Bracket Types"

uv run python brackets.py --self-check          # 21 checks, no dependencies
uv run --with pytest pytest -q                  # 138 tests

uv run python brackets.py --spec c src/*.c
echo '{"a": [1, 2}' | uv run python brackets.py --spec json
uv run python brackets.py --spec python --stream huge_file.py
uv run python brackets.py --spec python --auto-close partial.py
```

Stdlib only — no third-party dependency at runtime, `pytest` only for the test
suite. Exit code is 1 when any input fails, so it drops straight into CI.

## Test suite

138 tests, including two that earn their keep more than the rest:

- **Differential fuzzing** — 20,000 random strings over `([{}])xy `, comparing
  against the deliberately naive ten-line validator, which is a *correct*
  oracle for the `plain` spec. Any disagreement is a bug in the general one.
- **Deletion fuzzing** — 2,000 generated balanced strings, each with one
  random delimiter deleted; every result must be rejected. This catches the
  failure mode that fuzzing for false positives misses.

Plus streaming equivalence at eight chunk sizes across six specs, and CLI exit
codes.
