import { describe, it, expect } from 'vitest';
import { STEPS, ageOn, defaultData, expiryValid, luhnValid, validateStep } from '../src/lib/schemas.js';

const valid = () => ({
  ...defaultData(),
  email: 'ada@example.com',
  username: 'ada_l',
  password: 'correct horse 9',
  confirmPassword: 'correct horse 9',
  fullName: 'Ada Lovelace',
  birthDate: '1990-05-01',
  country: 'India',
  interests: ['data'],
  plan: 'free',
  acceptTerms: true,
});

describe('luhnValid', () => {
  it('accepts known-good test card numbers', () => {
    expect(luhnValid('4242424242424242')).toBe(true);
    expect(luhnValid('378282246310005')).toBe(true);
  });
  it('rejects a transposed digit and non-digits', () => {
    expect(luhnValid('4242424242424241')).toBe(false);
    expect(luhnValid('4242-4242')).toBe(false);
  });
});

describe('ageOn', () => {
  const today = new Date(Date.UTC(2026, 5, 15));
  it('counts whole years and respects the birthday boundary', () => {
    expect(ageOn('2013-06-15', today)).toBe(13);
    expect(ageOn('2013-06-16', today)).toBe(12);
  });
  it('rejects impossible dates', () => {
    expect(ageOn('2023-02-30', today)).toBeNaN();
    expect(ageOn('nope', today)).toBeNaN();
  });
});

describe('expiryValid', () => {
  const today = new Date(Date.UTC(2026, 5, 15));
  it('is valid through the end of the expiry month', () => {
    expect(expiryValid('06/26', today)).toBe(true);
    expect(expiryValid('05/26', today)).toBe(false);
    expect(expiryValid('01/27', today)).toBe(true);
  });
  it('rejects malformed values', () => {
    expect(expiryValid('13/30', today)).toBe(false);
    expect(expiryValid('6/30', today)).toBe(false);
  });
});

describe('account step', () => {
  it('passes with valid data', () => {
    expect(validateStep(0, valid()).ok).toBe(true);
  });
  it('reports one message per field, required first', () => {
    const { ok, errors } = validateStep(0, defaultData());
    expect(ok).toBe(false);
    expect(errors.email).toBe('Email is required');
    expect(errors.password).toBe('Password is required');
  });
  it('checks password strength rules', () => {
    expect(validateStep(0, { ...valid(), password: 'short1', confirmPassword: 'short1' }).errors.password).toMatch(/at least 10/);
    expect(validateStep(0, { ...valid(), password: 'onlyletterslong', confirmPassword: 'onlyletterslong' }).errors.password).toMatch(/number/);
  });
  it('cross-field: confirm must match, error lands on confirmPassword', () => {
    const { errors } = validateStep(0, { ...valid(), confirmPassword: 'different 12345' });
    expect(errors).toEqual({ confirmPassword: 'Passwords do not match' });
  });
  it('trims and validates email/username', () => {
    expect(validateStep(0, { ...valid(), email: '  ada@example.com  ' }).ok).toBe(true);
    expect(validateStep(0, { ...valid(), email: 'ada@' }).errors.email).toMatch(/valid email/);
    expect(validateStep(0, { ...valid(), username: 'a b' }).errors.username).toMatch(/letters, numbers/);
  });
});

describe('profile step', () => {
  it('rejects under-age and future birth dates', () => {
    const year = new Date().getUTCFullYear();
    expect(validateStep(1, { ...valid(), birthDate: `${year - 5}-01-01` }).errors.birthDate).toMatch(/at least 13/);
    expect(validateStep(1, { ...valid(), birthDate: `${year + 1}-01-01` }).errors.birthDate).toMatch(/future/);
  });
  it('phone is optional but validated when present', () => {
    expect(validateStep(1, { ...valid(), phone: '' }).ok).toBe(true);
    expect(validateStep(1, { ...valid(), phone: '+91 98765 43210' }).ok).toBe(true);
    expect(validateStep(1, { ...valid(), phone: 'call me' }).errors.phone).toBeTruthy();
  });
});

describe('preferences step (conditional)', () => {
  it('needs at least one interest', () => {
    expect(validateStep(2, { ...valid(), interests: [] }).errors.interests).toMatch(/at least one/);
  });
  it('frequency is required only when the newsletter is on', () => {
    expect(validateStep(2, { ...valid(), newsletter: false, frequency: '' }).ok).toBe(true);
    expect(validateStep(2, { ...valid(), newsletter: true, frequency: '' }).errors.frequency).toBeTruthy();
    expect(validateStep(2, { ...valid(), newsletter: true, frequency: 'weekly' }).ok).toBe(true);
  });
});

describe('plan step (conditional)', () => {
  const card = { cardNumber: '4242 4242 4242 4242', expiry: '12/99', cvc: '123' };
  it('free needs nothing else', () => {
    expect(validateStep(3, valid()).ok).toBe(true);
  });
  it('pro requires card details', () => {
    const { errors } = validateStep(3, { ...valid(), plan: 'pro' });
    expect(Object.keys(errors).sort()).toEqual(['cardNumber', 'cvc', 'expiry']);
    expect(validateStep(3, { ...valid(), plan: 'pro', ...card }).ok).toBe(true);
  });
  it('rejects a card failing Luhn and an expired card', () => {
    expect(validateStep(3, { ...valid(), plan: 'pro', ...card, cardNumber: '4242 4242 4242 4241' }).errors.cardNumber).toBeTruthy();
    expect(validateStep(3, { ...valid(), plan: 'pro', ...card, expiry: '01/20' }).errors.expiry).toMatch(/expired/);
  });
  it('team also needs company and 2-50 seats', () => {
    const base = { ...valid(), plan: 'team', ...card };
    expect(Object.keys(validateStep(3, base).errors).sort()).toEqual(['company', 'seats']);
    expect(validateStep(3, { ...base, company: 'Acme', seats: '1' }).errors.seats).toBeTruthy();
    expect(validateStep(3, { ...base, company: 'Acme', seats: '2.5' }).errors.seats).toBeTruthy();
    expect(validateStep(3, { ...base, company: 'Acme', seats: '10' }).ok).toBe(true);
  });
  it('unknown plan is rejected', () => {
    expect(validateStep(3, { ...valid(), plan: 'enterprise' }).errors.plan).toBe('Choose a plan');
  });
});

describe('review step', () => {
  it('requires accepting the terms', () => {
    expect(validateStep(4, { ...valid(), acceptTerms: false }).errors.acceptTerms).toMatch(/accept/);
    expect(validateStep(4, valid()).ok).toBe(true);
  });
});

describe('STEPS', () => {
  it('every step owns fields that exist in the default data', () => {
    const keys = Object.keys(defaultData());
    for (const s of STEPS) for (const f of s.fields) expect(keys).toContain(f);
  });
});
