const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const ATTENDING_VALUES = new Set(['yes', 'no', 'maybe']);
const MAX_NAME_LENGTH = 100;
const MAX_EMAIL_LENGTH = 254;
const MAX_MESSAGE_LENGTH = 500;
const MAX_GUEST_COUNT = 10;

/**
 * Validates and normalizes a raw RSVP submission body.
 * Returns { ok: true, value } on success or { ok: false, error } on failure.
 * Pure/DOM-free so it can be unit tested directly, independent of the HTTP layer.
 */
export function validateRsvp(body) {
  if (typeof body !== 'object' || body === null || Array.isArray(body)) {
    return { ok: false, error: 'Body must be a JSON object' };
  }

  const name = typeof body.name === 'string' ? body.name.trim() : '';
  if (!name || name.length > MAX_NAME_LENGTH) {
    return {
      ok: false,
      error: `name is required and must be at most ${MAX_NAME_LENGTH} characters`,
    };
  }

  const email = typeof body.email === 'string' ? body.email.trim() : '';
  if (!EMAIL_RE.test(email) || email.length > MAX_EMAIL_LENGTH) {
    return { ok: false, error: 'email is required and must be a valid address' };
  }

  if (!ATTENDING_VALUES.has(body.attending)) {
    return { ok: false, error: 'attending must be one of "yes", "no", "maybe"' };
  }

  const guestCount = Number(body.guestCount ?? 0);
  if (!Number.isInteger(guestCount) || guestCount < 0 || guestCount > MAX_GUEST_COUNT) {
    return {
      ok: false,
      error: `guestCount must be an integer between 0 and ${MAX_GUEST_COUNT}`,
    };
  }

  const message = typeof body.message === 'string' ? body.message.trim() : '';
  if (message.length > MAX_MESSAGE_LENGTH) {
    return { ok: false, error: `message must be at most ${MAX_MESSAGE_LENGTH} characters` };
  }

  return {
    ok: true,
    value: { name, email, attending: body.attending, guestCount, message },
  };
}
