/**
 * Pure, DOM-free string helpers for SMIL-animated icon markup.
 *
 * SMIL animations don't respect `prefers-reduced-motion` on their own, and
 * flipping an already-parsed `<animate>` element's `begin` attribute via
 * `setAttribute` after the document has loaded is unreliable across browser
 * SMIL implementations. The robust fix is to never insert the animation
 * elements into the DOM in the first place when reduced motion is requested
 * — `stripSmilAnimations` does that at the markup-string level, before
 * `gallery.js` ever calls `innerHTML`.
 */

/** Remove every `<animateMotion>…</animateMotion>` block (and its nested `<mpath>`) from `markup`. */
function stripAnimateMotion(markup) {
  return markup.replace(/<animateMotion\b[\s\S]*?<\/animateMotion>/gi, '');
}

/** Remove every self-closing `<animate>`/`<animateTransform>` tag from `markup`. */
function stripSelfClosingAnimate(markup) {
  return markup.replace(/<animate(?:Transform)?\b[^>]*\/>/gi, '');
}

/**
 * Strip all SMIL animation elements (`animate`, `animateTransform`,
 * `animateMotion`) out of an SVG markup string, leaving the static shape
 * intact. Used to render icons motionless under `prefers-reduced-motion`.
 */
export function stripSmilAnimations(markup) {
  return stripSelfClosingAnimate(stripAnimateMotion(markup));
}

/** Count of `<animate`, `<animateTransform`, or `<animateMotion` tags present in `markup`. */
export function countSmilAnimations(markup) {
  const matches = markup.match(/<animate(?:Transform|Motion)?\b/gi);
  return matches ? matches.length : 0;
}
