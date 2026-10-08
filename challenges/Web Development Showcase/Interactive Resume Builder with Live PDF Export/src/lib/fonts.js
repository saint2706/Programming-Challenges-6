/**
 * The three self-hosted font pairs and the plumbing that makes measuring trustworthy: block heights
 * are only correct once the fonts they were laid out in have loaded, so `ensureFonts` waits for
 * them before the first measurement and `onFontsLoaded` tells the app to re-measure when a font
 * face arrives later (for example when a typed character needs another Unicode subset).
 */

const SANS = 'system-ui, sans-serif';
const SERIF = 'Georgia, serif';

export const FONT_PAIRS = {
  inter: {
    id: 'inter',
    name: 'Inter',
    heading: { family: 'Inter', fallback: SANS },
    body: { family: 'Inter', fallback: SANS },
  },
  serif: {
    id: 'serif',
    name: 'Source Serif 4 + Inter',
    heading: { family: 'Source Serif 4', fallback: SERIF },
    body: { family: 'Inter', fallback: SANS },
  },
  plex: {
    id: 'plex',
    name: 'IBM Plex Sans',
    heading: { family: 'IBM Plex Sans', fallback: SANS },
    body: { family: 'IBM Plex Sans', fallback: SANS },
  },
};

const pairOf = (id) => FONT_PAIRS[id] ?? FONT_PAIRS.inter;

/** The distinct font families a pair needs, heading first. */
export function fontFamilies(pairId) {
  const pair = pairOf(pairId);
  return [...new Set([pair.heading.family, pair.body.family])];
}

/** CSS custom properties for a pair. */
export function fontVars(pairId) {
  const { heading, body } = pairOf(pairId);
  return {
    '--font-heading': `"${heading.family}", ${heading.fallback}`,
    '--font-body': `"${body.family}", ${body.fallback}`,
  };
}

// the four static weights the templates use (body, headline, entry titles, headings)
const WEIGHTS = [400, 500, 600, 700];
const MAX_SAMPLE = 4000;

/**
 * Load every face of the pair that `text` needs. Never rejects: a font that fails to load simply
 * means the fallback is measured and printed, consistently.
 */
export async function ensureFonts(pairId, text, fontSet = globalThis.document?.fonts ?? null) {
  if (!fontSet) return;
  const sample = String(text ?? '').slice(0, MAX_SAMPLE);
  const loads = fontFamilies(pairId).flatMap((family) =>
    WEIGHTS.map((weight) => fontSet.load(`${weight} 16px "${family}"`, sample)),
  );
  await Promise.allSettled(loads);
  try {
    await fontSet.ready;
  } catch {
    // a rejected ready promise just means "measure with what we have"
  }
}

/** Call `callback` whenever the browser finishes loading font faces. Returns an unsubscribe. */
export function onFontsLoaded(callback, fontSet = globalThis.document?.fonts ?? null) {
  if (!fontSet) return () => {};
  const handler = () => callback();
  fontSet.addEventListener('loadingdone', handler);
  return () => fontSet.removeEventListener('loadingdone', handler);
}
