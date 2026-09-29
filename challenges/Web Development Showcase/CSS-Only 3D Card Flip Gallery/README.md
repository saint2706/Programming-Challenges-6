# CSS-Only 3D Card Flip Gallery

**Category:** Web Development Showcase
**Difficulty:** B

**Status:** Implemented (JavaScript)

One HTML file + one CSS file. No JavaScript, no build step. Open it in a browser,
hover over cards to see the 3D flip effect. Each card front shows a project title and tags;
flip to reveal the description, star count (where applicable), and a GitHub link.

## What it does

- **8 real GitHub projects** featured as individual cards: Coding-For-MBA, AI Attendance
  Management System, Client Modding Guide, Scroll of Dharma, VITable, Geralt Discord
  Bot, VIT Academics Enhancer, and Python Course (GIM BDA) — all from the repo owner's
  actual GitHub portfolio.
- **3D perspective transforms** via pure CSS (`transform: rotateY(180deg)`,
  `transform-style: preserve-3d`, and `backface-visibility: hidden`) — no
  WebGL, no JavaScript, no library needed.
- **Hover to flip**: moving your mouse over any card rotates it 180 degrees
  to reveal the back face with the project description and a GitHub link.
- **Keyboard accessible flip**: `tabindex="0"` on each card lets you navigate
  with the keyboard. When a card receives focus (`:focus-within`), it flips
  the same as on hover, plus a red outline appears so keyboard users know
  which card is focused.
- **Responsive grid layout** using CSS Grid `repeat(auto-fill, minmax(...))`,
  so cards reflow smoothly from 1 to 4 columns depending on viewport width.
- **Fallback for motion-sensitive users**: `@media (prefers-reduced-motion: reduce)`
  replaces the 3D rotation with a simple CSS opacity crossfade (front fades out,
  back fades in), removing the vestibular-motion challenge while preserving the
  core interaction.

## Design notes

**The 3D flip mechanism**: Each `.card` has a `.card-inner` container with two
child elements: `.card-front` and `.card-back`. The container sets
`transform-style: preserve-3d` so child transforms happen in 3D space. Both faces
set `backface-visibility: hidden` so the side facing away from the viewer becomes
invisible. The `.card-back` starts rotated 180 degrees (`transform: rotateY(180deg)`);
on hover or focus, the `.card-inner` itself rotates 180 degrees, revealing the back
and hiding the front — a clean, hardware-accelerated effect supported across all
modern browsers.

**Keyboard accessibility**: The key decision here was using `:focus-within` to
trigger the flip when the card element itself is focused (via `tabindex="0"`),
not just when a descendant is focused. This means tabbing onto a card flips it
immediately, giving keyboard users visual feedback and the same dynamic experience
as mouse users. The focus outline (`outline: 3px solid red; outline-offset: 2px`)
is drawn *outside* the card so it doesn't get hidden by the back face.

**Reduced motion fallback**: The `@media (prefers-reduced-motion: reduce)` block
disables the 3D transform transition and instead applies an opacity transition to
both faces. The back face starts at `opacity: 0` with `pointer-events: none` (so
links on the back are not clickable until revealed). On hover/focus, the front
fades out and the back fades in — the interaction is preserved, the geometry is
stable, and vestibular-sensitive users get smooth opacity changes rather than
spatial rotation.

**Project data**: All 8 project titles, descriptions, and GitHub URLs are real
projects from the repo owner's GitHub account (`saint2706`). Star counts (69 for
Coding-For-MBA, 75 for Client Modding Guide) are current as of the time this
challenge was written; they are not auto-updated and are for display only.

**No JavaScript**: There is no `<script>` tag, no event listeners, no DOM
manipulation. The flip, focus styles, and responsive layout are purely CSS.
This makes the page instantly interactive, cacheable, and trivial to serve
over plain HTTP or `file://`.

## Run it

```bash
cd "challenges/Web Development Showcase/CSS-Only 3D Card Flip Gallery"

# Simply open the file directly in your browser:
# - On macOS: open index.html
# - On Windows: start index.html
# - On Linux: xdg-open index.html
# OR drag index.html into any open browser tab.

# To test keyboard navigation:
# 1. Tab through the cards (focus outline appears).
# 2. Focused cards flip the same way hovered ones do.
# 3. Press Tab to navigate among the GitHub links on the back.

# To test reduced-motion:
# macOS: System Settings > Accessibility > Display > Reduce motion
# Windows: Settings > Ease of Access > Display > Show animations
# Then refresh the page and hover a card — it should fade instead of rotate.
```

## Browser support

Tested and works in:

- Chrome/Edge 88+ (transform-style: preserve-3d, backface-visibility)
- Firefox 70+
- Safari 12+ (with `-webkit-` prefix for backface-visibility, included)

The transform and perspective properties are well-supported; the reduced-motion
fallback ensures graceful degradation on older browsers or accessibility-conscious setups.
