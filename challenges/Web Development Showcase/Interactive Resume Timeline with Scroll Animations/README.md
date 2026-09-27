# Interactive Resume Timeline with Scroll Animations

**Category:** Web Development Showcase
**Difficulty:** B

**Status:** Implemented (JavaScript)

No backend, no build step, no framework. Open `index.html`, scroll, and each
timeline entry reveals itself as it enters the viewport.

## What it does

- **Renders a real timeline** — education, work experience, and
  certifications, sorted reverse-chronologically and tagged by category
  (color-coded badges) — from a plain data module (`data.js`), not
  hand-written per-item markup.
- **Scroll-triggered reveals**, layered two ways (see Design notes): native
  CSS Scroll-Driven Animations where the browser supports them, a real
  `IntersectionObserver` everywhere else.
- **Respects `prefers-reduced-motion`** — reveals happen instantly with no
  motion for anyone who's asked for it.
- **Keyboard/AT-friendly markup** — a real `<ol>`/`<li>` list and `<time>`
  elements, not `<div>` soup.

## Design notes

**Why two reveal mechanisms:** the brief asks for "scroll-triggered reveals
via Intersection Observer", which is the correct universally-supported
choice — but 2026's actual modern default for this exact effect is native
CSS Scroll-Driven Animations (`animation-timeline: view()`), which needs no
JavaScript at all and runs on the compositor thread instead of the main
thread. Global support is roughly 84% (Chrome/Edge 115+, Firefox 132+,
Safari 18+) as of mid-2026, so it's used as the *primary* mechanism behind
an `@supports (animation-timeline: view())` block, with the
`IntersectionObserver` implementation in `reveal.js` as the fallback for
anything older. `reveal.js` feature-detects via `CSS.supports(...)` and, if
the native path is available, does nothing at all — it never creates a
duplicate observer, so the two mechanisms can't fight each other.

**Why the data/render/reveal split:** `data.js` is pure data, `render.js` is
a pure function (entries in, DOM nodes out, `document` passed as a
parameter rather than read as a global) and `reveal.js` takes its
dependencies (`supportsScrollTimeline`, `IntersectionObserverImpl`) as
injectable parameters. That's what makes both branches of the reveal logic
— "native CSS handles it, do nothing" and "attach a real observer, toggle a
class, unobserve once revealed" — unit-testable without a real browser.

**Where the timeline content comes from:** this is the user's actual
education/experience/certification history (Goa Institute of Management,
Vellore Institute of Technology, TheSmartBridge, Mood Indigo IIT Bombay,
AWS/IBM/freeCodeCamp/Coursera certifications), pulled from the source data
behind their live portfolio at `saint2706.github.io`. Contact details
(email, phone) are deliberately **not** included — the header links out to
GitHub and LinkedIn instead, since committing raw personal contact info
into a public challenge repo isn't worth the tradeoff for a demo project.

## Run it

```bash
cd "challenges/Web Development Showcase/Interactive Resume Timeline with Scroll Animations"

# View it — just open the file, no server or build step needed:
# open index.html directly in a browser

# Run the test suite:
npm install
npm test
```

## Where this is actually used

This exact pattern — a vertical timeline that reveals items on scroll — is
the shape of LinkedIn's "Experience" section on profile pages, most
developer-portfolio "About" pages, and product-changelog pages. The
native-CSS-first / IntersectionObserver-fallback layering mirrors how sites
are migrating off JS scroll libraries (GSAP ScrollTrigger, AOS, etc.) now
that the browser can do the same job natively for the common cases.
