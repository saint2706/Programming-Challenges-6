# Interactive Resume Builder with Live PDF Export

**Category:** Web Development Showcase
**Difficulty:** I

**Status:** Implemented (JavaScript)

A resume editor where the page you see is the page that prints. You edit in a side panel or type
straight onto the page, the preview is real A4 or Letter pages that update as you type, and Export
saves a PDF whose page count, page size, text and margins are the same as the preview. That claim
is tested, not assumed: a suite drives a real Chrome, saves a PDF, reads it back and compares.

## Run it

```bash
cd "challenges/Web Development Showcase/Interactive Resume Builder with Live PDF Export"
npm install
npm run dev          # the app, with hot reload
npm run build        # production build into dist/
npm test             # unit and component tests (jsdom), 379 tests
npm run test:e2e     # builds, then runs the real-Chrome PDF tests, 29 tests
npm run bench        # builds, then times repagination and typing in a real Chrome
```

The browser tests drive an installed Chrome or Edge (no browser is downloaded). Set `CHROME_PATH`
to point at one; without any browser they skip with a message instead of failing. Export uses the
browser's own "Save as PDF" in the print dialog, so there is no server and no PDF library.

## What it does

- **Hybrid editing.** The side panel holds structure and style: reorder sections (drag the handle,
  or Alt plus Up/Down, or the move buttons; each move is announced to screen readers), show or hide
  them, add and reorder entries and bullets, pick a template, page size, font pair, accent colour
  (with a contrast warning) and text size, and **Fit to N pages**. The words on the page (name,
  headline, summary, titles, bullets, skills) are also editable in place, as plain text; links,
  dates, location and contact details are edited in the side panel.
- **Three templates.** Classic (one column), Sidebar (two columns that paginate independently) and
  Compact (dense, one column).
- **Contact links that work in the PDF.** Email and phone are edited in the side panel, checked as
  you type, and printed as live `mailto:` and `tel:` links; a test opens the saved PDF and reads
  the links back.
- **JSON Resume.** The document is a [JSON Resume](https://jsonresume.org) subset with the
  builder's settings under `x-layout`, so import and export interoperate. Import is validated as a
  whole and never half-applied; one step of undo follows any replace.
- **Resilient.** Autosave to `localStorage` is versioned and validated on load; corrupt data is
  moved to a backup key, storage being disabled is reported and the app still works, and every
  link is checked against an allow-list (`http`, `https`, `mailto`, `tel`).

The starting resume is the author's real education, experience, projects and skills, taken from
their public portfolio. Email and phone are left empty (a test enforces it); the header links to
GitHub and LinkedIn instead. **Start over** loads a long fictional example (3+ pages) for trying
out pagination.

## How "print-accurate" works

The browser never decides where a page breaks at print time. A pure function does, and the DOM that
gets printed is already split into pages.

1. `blocks.js` turns the resume into regions of blocks (a heading, an entry header, a paragraph, a
   bullet list). Headings and entry headers are *keep with next*; a bullet list may split between
   bullets but never leaves fewer than two on either side of a break.
2. `measure.js` measures each block in a hidden copy of the page (same width, CSS and fonts) and
   caches the height by content, so a keystroke re-measures one block.
3. `paginate.js` places the blocks into fixed-height pages. It has no DOM, no randomness and no
   clock, so it is unit-tested directly, including property tests over random resumes.
4. The preview renders those pages as fixed-size boxes; `@page` and `break-after` make print
   identical. Pagination is coalesced to one pass per frame and flushed synchronously before print.

Two regions (the Sidebar template) paginate independently; the page count is the longer one.

## What a real Chrome caught

Four bugs showed up only when a PDF was made by a real Chrome and read back, not in any unit test.
Each is now pinned by a test in `e2e/print-accuracy.e2e.test.js`.

| Bug                                                                                                                                                                                    | How it showed up                                                                                                                              | Fix                                                                                                                                                                     |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **List spacing under-counted.** A bullet list's container has its own padding, but the paginator summed only the rows.                                                                 | Later pages ran about 20 px into the bottom margin.                                                                                           | Measure each list's own space ("chrome") and charge it once per page fragment. A test now checks every block renders exactly as tall as it was measured.                |
| **Scrambled PDF text order.** Bullets were `position: relative`, and Chrome paints positioned elements in a later pass.                                                                | The PDF's text lengths matched the preview, but all job headers came first and every bullet last: unreadable to an applicant tracking system. | Bullets are a flex row with a drawn dot. Nothing on a page is positioned, so paint order is document order. PDF text now equals preview text on every page.             |
| **Variable fonts become Type 3.** Chrome embeds a variable font as glyph-drawing outlines, not a TrueType subset.                                                                      | `isType3Font: true` for every font in the PDF.                                                                                                | Self-host static Fontsource weights. The PDF now embeds `Inter-Regular`, `-Medium`, `-SemiBold`, `-Bold` as real subsets.                                               |
| **Print media collapsed the preview.** Switching to print makes Chrome fire a font-loading event, which re-measured while print CSS hid the measuring tree: every height read as zero. | A 4-page resume became 1 page the moment print styles applied.                                                                                | The measuring tree stays laid out in print, a measure of a hidden tree throws instead of caching zeros, and nothing repaginates between `beforeprint` and `afterprint`. |

## Verification

| Check                                                       | Result                                                                                 |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Unit and component tests                                    | 379 passing                                                                            |
| Real-Chrome tests                                           | 29 passing; the 23 print and editing tests also passed on three runs in a row          |
| PDF page count and page size equal the preview              | 3 templates x A4 and Letter                                                            |
| PDF text per page equals the preview text per page          | all of the above, plus long unbroken URLs, accented, Greek, Cyrillic and Japanese text |
| All text inside the page margins                            | all of the above, and the longest valid email address                                  |
| Email and phone are live `mailto:` and `tel:` links in PDF  | 3 templates, typed in the side panel, and the longest valid address                    |
| Fonts embedded as real subsets, not Type 3                  | all three font pairs                                                                   |
| Caret stays in the field when it moves to the next page     | typed through a page boundary                                                          |
| axe-core                                                    | no violations, light and dark themes                                                   |
| No sideways scrolling at 390 px, same page count as desktop | passing                                                                                |

Not covered: the operating system's own print dialog (the tests save the PDF through Chrome's
print-to-PDF, which uses the same print layout), Firefox and Safari (print layout and font
embedding differ; only Chromium is tested), and DOCX or server-side PDF (out of scope).

### Speed

One run of `npm run bench` on the author's laptop (Chrome 154, 30 runs each, 120 keystrokes);
numbers vary by machine and from run to run.

| Resume                     | Pages | Blocks | Cold repagination p50 / p95 | Cached p50 / p95 |
| -------------------------- | ----- | ------ | --------------------------- | ---------------- |
| My resume (Sidebar)        | 1     | 35     | 2.9 / 5.0 ms                | 0.1 / 0.2 ms     |
| Long example (Compact)     | 3     | 66     | 7.4 / 11.6 ms               | 0.2 / 0.2 ms     |
| Long example x5 (Classic)  | 11    | 128    | 13.9 / 24.1 ms              | 0.3 / 0.4 ms     |
| Long example x16 (Classic) | 29    | 301    | 44.6 / 82.0 ms              | 0.6 / 1.0 ms     |

Typing in a bullet on page 1 repaints in about one frame (p50 14.1 ms at 3 pages, 14.3 ms at 11,
18.0 ms at 29; p95 15.0, 15.1 and 22.5 ms) and re-measures exactly one block per keystroke.

## Limits and choices

- **Plain text only.** No bold or italic, no photo: a photo hurts applicant tracking systems and
  rich text would need its own model (see the separate rich text editor challenge).
- **Contact details are checked strictly.** An address with a query (`?bcc=`), fragment, percent
  escape, comma, quote or control character is refused, and so is a phone with letters, a `tel:`
  prefix or `;ext=`. An import that carries one is rejected whole. Anything that gets past the
  schema (a hand-edited save) is shown as plain text, never as a link.
- **Write phone numbers in international form** (`+44 20 7946 0958`). A bracketed trunk zero such
  as `+44 (0)20 ...` is kept in the link and would not dial.
- **Dates are validated** as `YYYY`, `YYYY-MM` or `YYYY-MM-DD` and shown formatted on the page, so
  they are edited in the side panel, not as free text in place.
- **A block taller than a whole page** (one enormous bullet) is clipped and flagged in the page
  count; it is the only case where content can leave the page.
- **Test hook.** Opening the app with `?e2e` exposes a small read-only `window.__resume` used by the
  browser tests and the benchmark.

## Layout

```
src/lib/        schema, model, blocks, paginate, measure, fit, fonts, storage, print, io, links, ...
src/templates/  classic, sidebar, compact (which regions and sections each has)
src/components/ Preview, Page, Block, Editable, and the editor panel components
src/styles/     preview.css (pages, print) and templates.css (the three designs)
tests/          unit and component tests
e2e/            real-Chrome tests, the PDF reader, and the benchmark
```
