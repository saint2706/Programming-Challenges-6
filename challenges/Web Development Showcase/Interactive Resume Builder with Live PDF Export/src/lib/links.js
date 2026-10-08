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

// An address is `local@domain.tld` made of letters, digits and marks of any script plus the few
// punctuation marks real addresses use. Everything that could change what a mailto: link does
// (? # % & , ; : < > " \ / and brackets) is out, so an address can never carry a subject, a Bcc
// or a second recipient into the link.
const LOCAL_PART = /^[\p{L}\p{N}\p{M}._+\-'=!$*~]+$/u;
const DOMAIN_LABEL = /^[\p{L}\p{N}\p{M}](?:[\p{L}\p{N}\p{M}-]*[\p{L}\p{N}\p{M}])?$/u;
const MAX_EMAIL = 254;
const MAX_LOCAL_PART = 64;

/** True for one plain email address, with no padding. */
export function isEmail(value) {
  if (typeof value !== 'string' || value === '' || value.length > MAX_EMAIL) return false;
  const parts = value.split('@');
  if (parts.length !== 2) return false;
  const [local, domain] = parts;
  if (local.length > MAX_LOCAL_PART || !LOCAL_PART.test(local)) return false;
  const labels = domain.split('.');
  return labels.length >= 2 && labels.every((label) => DOMAIN_LABEL.test(label));
}

/** `mailto:` link for an address (padding ignored), or null when it is not a plain address. */
export function mailHref(value) {
  if (typeof value !== 'string') return null;
  const address = value.trim();
  return isEmail(address) ? `mailto:${address}` : null;
}

const PHONE_CHARS = /^[0-9 +\-().]*$/;

/** `tel:` link with only the dialable characters (a leading + and the digits), or null. */
export function telHref(value) {
  if (typeof value !== 'string') return null;
  const text = value.trim();
  if (!PHONE_CHARS.test(text)) return null;
  const digits = text.replace(/\D/g, '');
  if (digits.length < 3) return null;
  return `tel:${text.startsWith('+') ? '+' : ''}${digits}`;
}
