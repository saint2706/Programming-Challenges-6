# Custom Cursor & Micro-Interaction Playground

**Category:** Web Development Showcase
**Difficulty:** B

**Status:** Implemented (JavaScript)

A gallery of hover/cursor micro-interactions, each a standalone, copy-pasteable
snippet (visible via a "View source" toggle on every panel). No build step —
open `index.html` directly.

## What it does

1. **Damped cursor-follow dot** — a dot + ring trail the real cursor with
   inertia, driven by GSAP's `quickTo`.
2. **Magnetic button** — nudges toward the cursor within a radius, springs
   back on `mouseleave`.
3. **Tilt-on-move card** — 3D `rotateX`/`rotateY` tilt from pointer position
   relative to the card's bounding rect.
4. **Spotlight card** — a `radial-gradient` highlight follows the cursor via
   CSS custom properties (`--x`/`--y`) updated on `mousemove`.
5. **Text-scramble on hover** — hovering/focusing text triggers a
   scramble-to-decode animation.
6. **Cursor trail** — throttled, fading particles spawn along the cursor's
   path inside a bounded stage.

All six run as dependency-free vanilla JS/CSS except #1, which is the one
effect allowed a dependency.

## Design notes

**Why GSAP, and only for the cursor-follower:** damped, frame-rate-independent
inertia (the classic "dot lags slightly behind the real cursor" feel) is
exactly the interaction `gsap.quickTo` was built for, and GSAP is the
field-standard tool for this specific effect in real portfolio/agency sites
(it went fully free in 2025/2026, removing the previous licensing friction).
The other five effects are simple enough that a dependency would be pure
overhead, so they stay hand-rolled. GSAP is loaded as a classic global
`<script>` (not an ES-module `import`) specifically so it can carry a real
Subresource Integrity hash — SRI only applies to `<script src>` tags, not
module-graph imports — pinned to an exact version:

```html
<script
  src="https://cdn.jsdelivr.net/npm/gsap@3.15.0/dist/gsap.min.js"
  integrity="sha384-XmJ9SoHtVOHoQUcKvFAzVXwdkKo1Ie3bhmSoIAkcdsHGaIrVJIkmozyq0FJeb/Ly"
  crossorigin="anonymous"
></script>
```

The hash was computed from the actual downloaded file
(`openssl dgst -sha384 -binary gsap.min.js | openssl base64 -A`), not guessed
or copied from an unrelated source — the same "verify, don't trust" approach
this repo uses elsewhere for pinned dependencies.

**Feature detection, not assumption:** `main.js` checks
`window.matchMedia('(hover: hover) and (pointer: fine)')` before ever
touching the custom cursor — on touch devices it's skipped entirely rather
than shipping a hidden/broken cursor replacement. Separately,
`prefers-reduced-motion: reduce` disables every *continuously re-animating*
effect (custom cursor, magnetic button, tilt, cursor trail) while leaving the
spotlight and scramble effects on, since neither of those involves a
persistent transform animation loop — they're one-shot/pointer-driven
state changes.

**Why the math lives in its own module:** `scripts/math.js` exports
`clamp`, `lerp`, `magneticOffset`, and `tiltAngles` as pure, DOM-free
functions, and `scripts/scramble.js` exports `scrambleStep` the same way (it
takes an injectable RNG for deterministic tests). Every other script
(`cursor.js`, `magnetic.js`, `tilt.js`, `spotlight.js`, `trail.js`) is a thin
DOM-wiring layer around that math — this is what actually gets unit tested;
GSAP's own animation behavior and real mouse events are out of scope for a
unit test and aren't mocked.

## Run it

```bash
cd "challenges/Web Development Showcase/Custom Cursor & Micro-Interaction Playground"

# Just view it — no install needed:
# open index.html directly in a browser

# Run the test suite:
npm install
npm test
```

**Windows gotcha:** this folder's name contains a literal `&`
("Cursor & Micro-Interaction"), and npm's auto-generated `.cmd` shim for
`vitest` breaks on Windows when the path it's invoked from contains `&`
(`cmd.exe` treats it as a command separator even inside the shim's quoted
`%~dp0` expansion), so `npm test` itself fails with a misleading
`MODULE_NOT_FOUND` error on Windows. If that happens, bypass the shim and
invoke vitest directly instead:

```bash
node ./node_modules/vitest/vitest.mjs run
```

`npm install` is unaffected — only the generated Windows script shim is.
