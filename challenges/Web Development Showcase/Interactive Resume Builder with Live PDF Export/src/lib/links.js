/**
 * Link safety. Resume content (including imported JSON) is untrusted, and links end up
 * as clickable anchors on screen and in the PDF, so only a short allow-list of schemes
 * is ever rendered as a link. Everything else is shown as plain text.
 */

const ALLOWED_PROTOCOLS = new Set(['http:', 'https:', 'mailto:', 'tel:']);
const MAX_LENGTH = 2048;

// ASCII control characters, spaces and C1 controls. None may appear *inside* a link:
// browsers strip tabs/newlines from URLs, so "java\tscript:" would otherwise slip through
// a naive prefix check.
const FORBIDDEN = /[\u0000-\u0020\u007f-\u009f]/;

/** The trimmed URL if it is safe to render as a link, otherwise null. */
export function safeHref(value) {
  if (typeof value !== 'string') return null;
  const url = value.trim();
  if (url === '' || url.length > MAX_LENGTH || FORBIDDEN.test(url)) return null;
  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    return null;
  }
  return ALLOWED_PROTOCOLS.has(parsed.protocol) ? url : null;
}

export function isSafeHref(value) {
  return safeHref(value) !== null;
}
