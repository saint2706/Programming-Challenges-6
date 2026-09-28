import { describe, expect, it } from 'vitest';
import { validateRsvp } from '../src/validate.js';

const valid = {
  name: 'Ada Lovelace',
  email: 'ada@example.com',
  attending: 'yes',
  guestCount: 1,
  message: 'Looking forward to it!',
};

describe('validateRsvp', () => {
  it('accepts a fully valid submission and trims whitespace', () => {
    const result = validateRsvp({ ...valid, name: '  Ada Lovelace  ' });
    expect(result).toEqual({
      ok: true,
      value: { ...valid, name: 'Ada Lovelace' },
    });
  });

  it('defaults guestCount to 0 when omitted', () => {
    const { guestCount, ...rest } = valid;
    const result = validateRsvp(rest);
    expect(result.ok).toBe(true);
    expect(result.value.guestCount).toBe(0);
  });

  it('rejects a non-object body', () => {
    expect(validateRsvp(null).ok).toBe(false);
    expect(validateRsvp('nope').ok).toBe(false);
    expect(validateRsvp([]).ok).toBe(false);
  });

  it('rejects a missing or blank name', () => {
    expect(validateRsvp({ ...valid, name: '' }).ok).toBe(false);
    expect(validateRsvp({ ...valid, name: '   ' }).ok).toBe(false);
    expect(validateRsvp({ ...valid, name: undefined }).ok).toBe(false);
  });

  it('rejects a name over 100 characters', () => {
    expect(validateRsvp({ ...valid, name: 'a'.repeat(101) }).ok).toBe(false);
  });

  it('rejects a malformed email', () => {
    for (const bad of ['not-an-email', 'missing@domain', '@nodomain.com', '']) {
      expect(validateRsvp({ ...valid, email: bad }).ok).toBe(false);
    }
  });

  it('rejects an attending value outside the enum', () => {
    for (const bad of ['sure', 'YES', '', undefined, null]) {
      expect(validateRsvp({ ...valid, attending: bad }).ok).toBe(false);
    }
  });

  it('rejects a non-integer or out-of-range guestCount', () => {
    for (const bad of [-1, 11, 1.5, 'three', NaN]) {
      expect(validateRsvp({ ...valid, guestCount: bad }).ok).toBe(false);
    }
  });

  it('accepts guestCount at the boundaries (0 and 10)', () => {
    expect(validateRsvp({ ...valid, guestCount: 0 }).ok).toBe(true);
    expect(validateRsvp({ ...valid, guestCount: 10 }).ok).toBe(true);
  });

  it('rejects a message over 500 characters', () => {
    expect(validateRsvp({ ...valid, message: 'a'.repeat(501) }).ok).toBe(false);
  });

  it('treats a missing message as an empty string', () => {
    const { message, ...rest } = valid;
    const result = validateRsvp(rest);
    expect(result.ok).toBe(true);
    expect(result.value.message).toBe('');
  });
});
