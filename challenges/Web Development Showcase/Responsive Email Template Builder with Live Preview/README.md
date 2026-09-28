# Responsive Email Template Builder with Live Preview

**Category:** Web Development Showcase
**Difficulty:** B

**Status:** Implemented (JavaScript/Svelte)

A block-based builder that composes a real, table-based HTML email (inline
styles only, bulletproof-button technique) and renders it live in a
sandboxed, width-toggleable preview frame.

## What it does

- **Block palette:** add heading, paragraph, image, button, divider, spacer,
  and two-column blocks. Click a block to edit its fields inline; reorder
  with ↑/↓ or remove it.
- **Email-safe serialization:** every block renders into nested
  `<table role="presentation">` markup with **inline styles only** — no
  `<style>` block, since many email clients (Gmail among them) strip
  `<head>` entirely. The button block uses the classic **bulletproof-button**
  pattern (a table-wrapped `<a>`, color/radius on the `<td>` background
  rather than relying on `<a>` padding, which Outlook's Word-based rendering
  engine handles inconsistently).
- **Live preview:** the generated HTML renders inside a `sandbox=""`
  `<iframe srcdoc="...">` — fully isolated from the builder app's own CSS,
  and scripts disabled since the frame only ever needs to display markup.
  A Desktop (600px) / Mobile (375px) width toggle resizes the frame, the
  standard way real email builders demonstrate responsive behavior without
  needing a second render target.
- **Copy HTML:** copies the final standalone document (the same string
  rendered in the preview) to the clipboard for pasting into a real ESP.

## Design notes

**Why Svelte, breaking this category's vanilla-JS streak:** challenges 1-3
in this category are deliberately framework-free. This one is the explicit
exception — the user asked for a more modern stack on this specific
challenge. Svelte 5 (runes: `$state`/`$derived`/`$props`) was chosen over
Vue 3 for the smaller compiled-away runtime (no virtual DOM, no framework
code shipped to the page) — a good fit for a small, self-contained demo app
scaffolded via Vite. `App.svelte` owns all builder state (`blocks`,
`selectedId`, `previewWidth`); `BlockEditor.svelte` and `Preview.svelte` are
presentational, driven entirely by callback props rather than
`createEventDispatcher` (Svelte 5's recommended pattern).

**Why serialization is a plain, framework-free module:** `src/lib/serialize.js`
takes a block array and returns an HTML string — no Svelte import, no DOM
API. This is the same "pure logic, thin UI wiring" split every other
challenge in this folder uses, and it's what actually gets unit-tested
directly (21 tests), independent of component rendering.

**Reordering via ↑/↓, not drag-and-drop:** drag-to-reorder is the explicit
subject of a *different* challenge in this category (#8, "Drag-to-Reorder
Photo Album with Masonry Layout"). Building a full custom drag
implementation here would duplicate that challenge's scope for no added
value to this one's actual brief ("table-based HTML email quirks, live
preview"), so reordering here is simple, keyboard-accessible move buttons.

**Security — escaping is manual and mandatory:** because the output is a
*string* of HTML (not DOM nodes), every user-typed value (heading/paragraph
text, image alt, button label) goes through `escapeHtml()` before
interpolation — there's no `textContent` safety net when you're building a
string. `href` values (button link, image `src`) are additionally checked by
`isSafeHref()`, which rejects `javascript:`, `vbscript:`, and `data:`
schemes — including whitespace/control-character-obfuscated variants like
`java\tscript:` — falling back to `#` rather than emitting the raw value.
This mirrors a real stored-XSS bug a past fork in this repo shipped by
skipping exactly this step.

**Config:** `vite.config.js` doubles as the Vitest config (`defineConfig`
from `vitest/config`, with the Svelte plugin) rather than a separate
`vitest.config.js`, since Vite is already required here for the Svelte
compile step — unlike the framework-free challenges in this folder, there's
no reason to keep them apart.

## Run it

```bash
cd "challenges/Web Development Showcase/Responsive Email Template Builder with Live Preview"
npm install

# Dev server (builder UI):
npm run dev

# Production build (outputs to dist/, gitignored):
npm run build

# Run the test suite (21 tests, the serialization/escaping/XSS logic):
npm test
```
