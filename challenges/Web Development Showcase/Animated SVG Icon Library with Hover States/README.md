# Animated SVG Icon Library with Hover States

**Category:** Web Development Showcase
**Difficulty:** B

**Status:** Implemented (JavaScript)

A gallery of 19 original, hand-coded SVG icons themed around data/analytics
work — database, charts, ETL, ML, and infra concepts — each with a distinct
hover animation. No build step — open `index.html` directly.

## What it does

- A filterable grid (`data`, `workflow`, `infra`, `ml`) of 19 icons, each on a
  24x24 `viewBox`, `stroke="currentColor"` (inherits color from its
  container), decorative (`aria-hidden="true"`) with the icon's name as the
  visible, accessible label on its panel.
- **14 icons animate via CSS** (`style.css`, hover/`:focus-visible`-scoped
  `@keyframes`/transitions): bars growing, a line drawing in, disks
  separating, a spinning gear, a blinking terminal cursor, and more.
- **5 icons animate via native SMIL** (`<animate>`/`<animateTransform>`/
  `<animateMotion>` inline in the SVG itself, not CSS) to genuinely
  demonstrate that technique rather than defaulting to CSS everywhere:
  `neural-network` (nodes pulse in sequence), `dashboard-gauge` (needle
  sweeps via `animateTransform`), `globe-locale` (meridians spin via
  `animateTransform`), `flask-experiment` (bubbles rise via `animate` on
  `cy`), `git-branch` (a commit dot travels along a path via
  `animateMotion`/`<mpath>`).
- Every card has a "View source" toggle with a **Copy SVG** button, so any
  icon can be lifted out as a standalone, self-contained snippet — same
  copy-pasteable-snippet UX as the Custom Cursor playground.

## Design notes

**Icon set, not a generic UI kit:** per an explicit choice made before
building this, the set is themed around the repo owner's actual background
(Data Storyteller & Analytics Strategist — Python/SQL/Pandas/Tableau/
ML/PyTorch/AWS, ETL pipelines, git) rather than a generic search/settings/
cart set — `database`, `bar-chart`, `line-chart`, `donut-chart`,
`scatter-plot`, `dataframe-table`, `magnifier-insight`, `funnel` (ETL),
`api-plug`, `notebook-pen`, `terminal`, `cpu-chip`, `cloud-aws`,
`settings-gear`, plus the 5 SMIL icons above. Every path is hand-drawn for
this challenge, not copied from an existing icon library, so there's no
licensing entanglement.

**Why the CSS/SMIL split is 14/5, not 50/50:** the brief says "CSS/SMIL
animation," and the honest read is "CSS is the default, SMIL exists and is
worth demonstrating" — not "half and half for its own sake." SMIL is
genuinely the *right* tool for a couple of these (`animateMotion` walking a
dot along an arbitrary path via `<mpath>` has no clean CSS equivalent; CSS
`offset-path` is closer but isn't the "native SMIL" technique the brief
calls out), so those get SMIL; everything expressible cleanly as a
transform/opacity/dasharray hover state stays CSS, which is simpler to
read, override, and reduced-motion-gate.

**The `prefers-reduced-motion` nuance that actually matters here:** CSS
animations are trivially gated with a media query
(`@media (prefers-reduced-motion: reduce) { .icon-panel * { animation: none
!important; transition: none !important; } }` in `style.css`). **SMIL
animations do not respect that media query at all** — they're a separate
timing model. The naive fix (query the `<animate>` elements after the page
loads and rewrite their `begin` attribute to `"indefinite"`) is unreliable:
SMIL's event-based timing is resolved when the element enters the document,
and mutating `begin` via `setAttribute` after that point doesn't reliably
re-register the event listener across browser SMIL implementations. The
robust fix, implemented in `scripts/smil.js` + `scripts/gallery.js`: decide
*once*, before any DOM insertion, whether the browser wants reduced motion
(`matchMedia('(prefers-reduced-motion: reduce)').matches`), and if so, strip
every `<animate>`/`<animateTransform>`/`<animateMotion>` element out of the
markup **string** (`stripSmilAnimations`) before it's ever inserted — so a
reduced-motion browser never sees an animated element in the first place,
rather than trying to stop one that's already running.

**What's actually unit-tested, and why:** consistent with this repo's
established pattern (test pure logic, don't mock real DOM/animation
behavior) — `scripts/icons.js` is a pure data module (no DOM access) and
`scripts/smil.js` is pure string manipulation, so both are fully covered:
every icon has required metadata, no duplicate ids, the CSS/SMIL split
stays in the 3–5-icon SMIL range the brief implies ("a handful"), every
SMIL-technique icon actually contains an `<animate*>` element (and no
CSS-technique icon accidentally does), every `<mpath href="#…">` reference
resolves to an id that exists in the same markup, and `stripSmilAnimations`
correctly removes self-closing `<animate>`/`<animateTransform>` tags and
paired `<animateMotion>…</animateMotion>` blocks (including the nested
`<mpath>`) without disturbing unrelated markup. `scripts/gallery.js` (DOM
rendering, the copy-to-clipboard wiring, the category filter) is the thin
DOM-wiring layer on top and isn't unit tested, same as `main.js`/`cursor.js`
etc. in the Custom Cursor playground.

**A safety note on `innerHTML` usage in `gallery.js`:** `stage.innerHTML =
iconMarkupForRender(icon)` inserts raw markup, which is normally an XSS red
flag — but `icon.markup` originates entirely from the static, hand-authored
`icons.js` data module in this same repo, never from user input, a network
response, or a URL parameter, so there's no untrusted data crossing that
boundary. Every other piece of user-visible text (`icon.name`, the
"View source" `<pre><code>` block, button labels) is set via `textContent`,
not `innerHTML`, specifically to avoid needing manual HTML-escaping.

## Run it

```bash
cd "challenges/Web Development Showcase/Animated SVG Icon Library with Hover States"

# Just view it — no install needed:
# open index.html directly in a browser

# Run the test suite:
npm install
npm test
```
