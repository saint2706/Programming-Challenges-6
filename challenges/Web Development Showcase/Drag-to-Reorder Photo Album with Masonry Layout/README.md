# Drag-to-Reorder Photo Album with Masonry Layout

**Category:** Web Development Showcase
**Difficulty:** I

**Status:** Implemented (JavaScript)

A photo album you can reorder by dragging, laid out as a masonry grid, whose
order survives a reload. Built with Vite, vanilla JS and Atlassian's
[Pragmatic drag and drop](https://atlassian.design/components/pragmatic-drag-and-drop/)
(a thin layer over native HTML5 drag events, about 5 kB), plus a full keyboard
alternative for people who can't or won't drag.

## Run it

```bash
cd "challenges/Web Development Showcase/Drag-to-Reorder Photo Album with Masonry Layout"
npm install
npm run dev      # http://localhost:5173
npm test         # 71 tests (vitest: node + jsdom)
npm run build    # static site in dist/ (base './', so it can be hosted from any path)
```

## How it works

| Module | Job |
| --- | --- |
| `src/order.js` | Pure, DOM-free order logic: `moveItem`, `nudge`, `placeRelativeTo`, `reconcile`, `serialize`/`deserialize`. Every function returns a new array. |
| `src/storage.js` | `localStorage` load/save/clear, every call guarded so a missing, full or throwing store degrades to "no persistence". |
| `src/masonry.js` | Turns each photo's aspect ratio into a CSS grid row span. |
| `src/album.js` | Builds the DOM, owns the order, keyboard reordering, FLIP animation, persistence and announcements. |
| `src/dnd.js` | Wires Pragmatic drag and drop onto the tiles; only ever calls `album.commitDrop`. |
| `src/photos.js` | The photo data: id, file, size, alt text, photographer, source URL. |

### Design notes

**Masonry without measuring images.** The grid is `display: grid` with
`grid-auto-rows: 4px`; each tile spans `ceil((columnWidth * height / width + gap) / 4)`
rows. Because every photo's width and height are declared up front, the span
is computed from the column width alone, so nothing waits on an image load and
the layout never shifts. A `ResizeObserver` recomputes the spans when the
column width changes. The trade-off is that reading order is row-major with
"shortest gap" packing done by the browser's auto-placement, not a
shortest-column-first algorithm, so columns can end slightly ragged. In exchange
the DOM order equals the visual reading order, which keeps keyboard order,
screen-reader order and the persisted order all the same thing. (CSS
`grid-template-rows: masonry` is still not shipping in stable browsers.)
`tests/photos.test.js` checks each declared size against the real WebP so a
wrong pair can't silently cause overlap. I also checked in a real browser that no two
tiles' rectangles overlap.

**Drop semantics are computed after removal.** Dropping on the left or right
half of a tile means "insert before" or "after" it. `placeRelativeTo` removes
the dragged item first and only then finds the insertion index, which makes
"drop on the near edge of my own neighbour" a true no-op instead of an
off-by-one shuffle. The closest-edge hitbox drives a drop indicator bar drawn
in the gap.

**Keyboard reordering is a first-class path, not an afterthought.** Focus a
tile, press <kbd>Space</kbd> or <kbd>Enter</kbd> to grab it, move it with the
arrow keys (<kbd>Home</kbd> and <kbd>End</kbd> jump to the ends), then
<kbd>Space</kbd> to drop or <kbd>Esc</kbd> to cancel back to the original order.
Focus leaving the tile drops it. Every step is announced through an
`aria-live="polite"` region ("Moved Golden petals. Position 2 of 15."), tiles
carry `aria-roledescription="sortable photo"` and an "N of M" label, and the
order is saved only on drop, so a cancelled reorder writes nothing.

**Persisted order is reconciled, never trusted.** Storage holds
`{"v":1,"order":[ids]}`. On load, `reconcile` drops unknown and duplicate ids,
ignores non-string entries, appends photos added since the save in canonical
order, and treats a wrong version or malformed JSON as "no saved order". So
corrupt storage, a removed photo and a newly added photo all just work.

**Motion.** Displaced tiles slide with a FLIP animation (Web Animations API);
with `prefers-reduced-motion: reduce` the animation is skipped in JS and all
CSS transitions are disabled.

**No untrusted HTML.** Photo titles, credits and alt text are set with
`textContent` and `setAttribute`, never `innerHTML`; a test feeds a photographer
name containing `<b>` and asserts it stays text.

## Tests

71 tests, all offline:

- `order.test.js`: the pure functions, including off-by-one drop cases and stale or corrupt saved data.
- `album.test.js` (jsdom): rendering, keyboard grab/move/drop/cancel, end stops, focus retention, announcements, persistence across "reloads", corrupt or throwing storage, reset.
- `masonry.test.js`: row-span maths (covers height plus gap, rounds up so tiles never overlap, degrades on bad input).
- `photos.test.js`: unique ids, non-empty credit fields, and each declared width/height equals the actual file's size.

I also drove the running app in Chrome: a real drag reordered the tiles and wrote
the new order to `localStorage`; a reload restored it; the keyboard flow
worked; no tiles overlapped.

## Photo sources and licences

All 15 photos are from [Unsplash](https://unsplash.com) and used under the
[Unsplash License](https://unsplash.com/license) (free to use, no permission
or attribution required; credit is shown anyway). I checked each photo's page
on Unsplash for the photographer and description; the credits below match.
The originals were saved as 1000 px-wide JPEGs and converted to WebP
(long edge at most 800 px) by `npm run photos`.

Two honest caveats: the JPEGs carry no EXIF or other embedded metadata, so I
could not verify pixel-for-pixel that each file is the image on the page it is
credited to (filenames and content match the page descriptions); and the
originals live in `raw/`, which is git-ignored, so `npm run photos` only works
if you re-download them there (the converted `public/photos/*.webp` files are
committed and are all the app needs).

| Photo | Photographer | Source |
| --- | --- | --- |
| Whangarei Falls footbridge | Tim Swaan | [eOpewngf68w](https://unsplash.com/photos/eOpewngf68w) |
| Golden petals | Volodymyr Lymariev | [wR9VG-W8nU4](https://unsplash.com/photos/wR9VG-W8nU4) |
| Reeds at dusk | Mark Dixon | [hLYbJB-D5Gg](https://unsplash.com/photos/hLYbJB-D5Gg) |
| Dragonfly | CR | [IG1yO9YDkqU](https://unsplash.com/photos/IG1yO9YDkqU) |
| Lightning storm | Marek Piwnicki | [d-p06WttJJE](https://unsplash.com/photos/d-p06WttJJE) |
| Bumblebee | Dmytro Koplyk | [a3HmolpGW3s](https://unsplash.com/photos/a3HmolpGW3s) |
| Shark Fin Cove | Karla Hernandez | [shn9z-172sM](https://unsplash.com/photos/shn9z-172sM) |
| Cat nap | Bastian Alexander-Coleman | [8CsDIpCytF0](https://unsplash.com/photos/8CsDIpCytF0) |
| Hibiscus | Iván Díaz | [hBfY_uyLwAE](https://unsplash.com/photos/hBfY_uyLwAE) |
| North America Nebula | Wallace Henry | [3lSdgBnv9ag](https://unsplash.com/photos/3lSdgBnv9ag) |
| Tortoise | Adrian Botica | [tI7TOjJOFqI](https://unsplash.com/photos/tI7TOjJOFqI) |
| Painted lady | Dmytro Koplyk | [R0PjjzRWWf8](https://unsplash.com/photos/R0PjjzRWWf8) |
| Bald eagle | Venti Views | [DTY3cKv0pvc](https://unsplash.com/photos/DTY3cKv0pvc) |
| Crescent moon | Xx M | [1o5hFRQ77l0](https://unsplash.com/photos/1o5hFRQ77l0) |
| Northern cardinal | Dmytro Koplyk | [TXmLf1NSTVs](https://unsplash.com/photos/TXmLf1NSTVs) |

## Limitations

- **Touch dragging.** Pragmatic drag and drop sits on native HTML5 drag events, which mobile browsers support unevenly (Safari on iOS supports them; some Android browsers need a long press). The keyboard path works everywhere, but there are no on-screen move buttons for touch-only users.
- **One album, one order.** The order lives in a single `localStorage` key, per browser; it is not synced anywhere.
- **Photos are a fixed list** in `src/photos.js`; there is no upload.
- **Ragged columns.** Row-span masonry keeps DOM order equal to reading order but does not balance column heights the way a shortest-column algorithm would.
