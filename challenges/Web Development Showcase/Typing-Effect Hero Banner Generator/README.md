# Typing-Effect Hero Banner Generator

**Category:** Web Development Showcase
**Difficulty:** B

**Status:** Implemented (JavaScript)

A configurable typing/deleting hero-banner text effect, built and shipped as a
genuinely **publishable npm package** (ESM + CJS + UMD bundles, generated
type declarations) rather than a copy-pasteable snippet — per the brief's
"ship as an npm-style snippet" requirement.

## What it does

`createTypingHero(el, options)` mounts a self-contained typing/deleting
animation into any DOM element: it cycles through an array of strings,
"typing" each one character at a time, pausing, deleting it, and moving on to
the next (optionally looping forever, or stopping after the last string).

```js
import { createTypingHero } from '@programming-challenges-6/typing-hero';

const hero = createTypingHero(document.getElementById('hero'), {
  strings: ['Build things.', 'Ship them.', 'Iterate.'],
  typingSpeed: 55, // ms per typed character
  deletingSpeed: 28, // ms per deleted character
  pauseBeforeDelete: 1400, // ms to hold a finished phrase
  pauseBeforeNext: 400, // ms to hold the empty state between phrases
  startDelay: 300, // ms before the first character types
  loop: true, // cycle forever; false stops after the last phrase
  cursorChar: '|',
  cursorBlink: true,
  onComplete: () => console.log('done'), // only fires when loop: false
});

hero.start();
// hero.stop();    // pause, resumable via start() again
// hero.destroy();  // stop + tear down the DOM it created
```

See [`demo/index.html`](./demo/index.html) for a full working page that
imports the **built** `dist/index.js` bundle directly.

## Design notes

**Why the state machine is a separate, pure module.** `src/stateMachine.js`
exports `createInitialState()` and `advance(state, options)` as plain
functions with no DOM, timers, or side effects — `advance` takes a state and
returns the next state plus how long to wait before the next step. This is
the entire "what should happen next" logic for the effect (typing → pause →
deleting → pause → next phrase, with looping/non-looping and empty-string
edge cases), and it's exactly what's unit tested: 15 assertions across every
phase transition with zero timers or a DOM involved. `src/index.js` is a thin
DOM/timer wiring layer around it (`setTimeout` loop calling `advance` and
re-rendering), which is what the smaller integration test suite covers using
`vi.useFakeTimers()`.

**Screen-reader handling, not an afterthought.** A letter-by-letter typing
effect is a known accessibility trap: naively updating live text on every
keystroke makes a screen reader announce every single character as it
appears. Here, the visual `<span>` doing the character-by-character reveal is
`aria-hidden="true"`, and a separate visually-hidden (`.typing-hero__sr-only`,
a real sr-only clip pattern — not `display: none`, which *would* be pulled
from the accessibility tree) element holds the **full current phrase**,
updated exactly once per phrase-change (`announcePhraseIfChanged` in
`src/index.js` only writes to it when `phraseIndex` actually changes), inside
an `aria-live="polite"` region. `prefers-reduced-motion: reduce` is checked
via `matchMedia` before `start()` ever schedules a timer — on a reduced-motion
system the first phrase renders statically and nothing animates at all,
rather than just running faster.

**Why `tsup` and why the package stays `"private": true` anyway.** `tsup` is
a zero-config esbuild-based bundler — a few lines of config
(`tsup.config.js`) produce real ESM (`dist/index.js`), CommonJS
(`dist/index.cjs`), and an IIFE global build (`dist/index.global.js`) for a
plain `<script>` tag, plus `.d.ts`/`.d.cts` declarations generated from this
package's JSDoc annotations (via `tsconfig.json`'s `allowJs`/`checkJs`/
`emitDeclarationOnly`, no TypeScript source required). `package.json`'s
`exports` map points consumers at the right build per `import`/`require`.
Despite being structurally ready for `npm publish`, the package keeps
`"private": true` deliberately — this is a portfolio-repo demo, not a
package actually intended for the public registry, and `private: true` is a
guard against any tool ever running `npm publish` here by accident, not a
sign the packaging is incomplete.

**Why the CSS ships unbundled.** `styles.css` at the package root (not run
through `tsup`) provides the cursor blink keyframe and the sr-only clip
pattern. It's plain CSS with no preprocessing step, so there's nothing for a
bundler to usefully do to it — it's exposed via the `"./style.css"` export
subpath and listed directly in `package.json`'s `files`, the same way many
real npm packages ship an unbundled companion stylesheet.

## Where this is actually used

**Hero banners.** A typing effect on a landing page is a common way to show several short
value statements in one slot. Typed.js is a published typing-effect library on npm that does
the same job, with a constructor that takes a selector and a list of strings. Check its
license file before copying code from it, since listings disagree on the license.

**A timer-driven state machine.** The engine is a small loop of states: type a character,
hold a finished phrase, delete, pause, move to the next phrase. Each state sets one timer.
Packaging it as ESM, CJS and UMD bundles is what lets the same file work in an npm project
and in a plain script tag.

**Where a demo falls short.** Screen readers can announce the typed text one character at a
time if the text sits in a live region. A common fix is to keep the full phrase in an
`aria-label` or a visually hidden element and mark the animated span `aria-hidden`. This
generator does not do that yet, and it also needs a reduced-motion path that shows the
final phrase with no animation.

## Run it

```bash
cd "challenges/Web Development Showcase/Typing-Effect Hero Banner Generator"

npm install
npm run build   # produces dist/index.js, dist/index.cjs, dist/index.global.js, dist/index.d.ts
npm test        # 20 tests: 15 pure state-machine, 5 DOM/timer integration

# View the demo (imports the real dist/ build, not src/):
# open demo/index.html directly in a browser after `npm run build`
```

`dist/` is generated output (gitignored via this folder's own `.gitignore`)
— always run `npm run build` before opening the demo or consuming the
package locally.
