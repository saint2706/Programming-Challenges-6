import { describe, it, expect } from 'vitest';
import { Wizard } from '../src/lib/wizard.svelte.js';

function fillAccount(w) {
  Object.assign(w.data, { email: 'ada@example.com', username: 'ada_l', password: 'correct horse 9', confirmPassword: 'correct horse 9' });
}
function fillAll(w) {
  fillAccount(w);
  Object.assign(w.data, { fullName: 'Ada', birthDate: '1990-05-01', country: 'India', interests: ['data'], acceptTerms: true });
}

describe('Wizard', () => {
  it('blocks Next on an invalid step and touches every field', () => {
    const w = new Wizard();
    expect(w.next()).toBe(false);
    expect(w.step).toBe(0);
    expect(Object.keys(w.errors)).toEqual(expect.arrayContaining(['email', 'password']));
    expect(w.touched.email).toBe(true);
    expect(w.verified).toEqual([]);
  });

  it('verifies and advances on a valid step, clearing errors', () => {
    const w = new Wizard();
    fillAccount(w);
    expect(w.next()).toBe(true);
    expect(w.step).toBe(1);
    expect(w.verified).toEqual([0]);
    expect(w.errors).toEqual({});
    expect(w.percent).toBe(20);
  });

  it('go() refuses to skip unverified steps', () => {
    const w = new Wizard();
    expect(w.go(3)).toBe(0);
    fillAccount(w);
    w.next();
    expect(w.go(4)).toBe(1);
    expect(w.back()).toBe(0);
  });

  it('editing a step un-verifies it, so later steps become unreachable again', () => {
    const w = new Wizard();
    fillAccount(w);
    w.next();
    w.go(0);
    w.data.email = 'new@example.com';
    w.edit('email');
    expect(w.verified).toEqual([]);
    expect(w.maxStep).toBe(0);
  });

  it('errors are silent until blur, then live', () => {
    const w = new Wizard();
    w.data.email = 'bad';
    w.edit('email');
    expect(w.errors).toEqual({});
    w.touch('email');
    expect(w.errors.email).toMatch(/valid email/);
    w.data.email = 'ok@example.com';
    w.edit('email');
    expect(w.errors.email).toBeUndefined();
  });

  it('submit succeeds only when every step is valid', () => {
    const w = new Wizard();
    fillAll(w);
    for (let i = 0; i < 4; i++) expect(w.next()).toBe(true);
    expect(w.step).toBe(4);
    expect(w.submit()).toEqual({ ok: true });
    expect(w.submitted).toBe(true);
  });

  it('submit after a resume jumps back to the step whose secrets were not restored', () => {
    const data = { ...new Wizard().data, email: 'ada@example.com', username: 'ada_l', fullName: 'Ada', birthDate: '1990-05-01', country: 'India', interests: ['data'], acceptTerms: true };
    const w = new Wizard({ data, step: 4, verified: [0, 1, 2, 3] });
    expect(w.step).toBe(4);
    expect(w.submit()).toEqual({ ok: false, failedStep: 0 });
    expect(w.step).toBe(0);
    expect(w.errors.password).toBeTruthy();
    expect(w.verified).toEqual([1, 2, 3]); // only the failing step lost its verification
  });

  it('after fixing the failed step, Next skips ahead over still-verified steps', () => {
    const data = { ...new Wizard().data, email: 'ada@example.com', username: 'ada_l', fullName: 'Ada', birthDate: '1990-05-01', country: 'India', interests: ['data'], plan: 'pro', acceptTerms: true };
    const w = new Wizard({ data, step: 4, verified: [0, 1, 2, 3] });
    expect(w.submit()).toEqual({ ok: false, failedStep: 0 });
    expect(w.verified).toEqual([1, 2]); // account and plan (no card restored) both failed
    Object.assign(w.data, { password: 'correct horse 9', confirmPassword: 'correct horse 9' });
    expect(w.next()).toBe(true);
    expect(w.step).toBe(3); // straight to the plan step, not step 2
    Object.assign(w.data, { cardNumber: '4242 4242 4242 4242', expiry: '12/99', cvc: '123' });
    expect(w.next()).toBe(true);
    expect(w.step).toBe(4);
    expect(w.submit()).toEqual({ ok: true });
  });

  it('a restored draft is clamped to what its verified list allows', () => {
    const w = new Wizard({ data: new Wizard().data, step: 4, verified: [0] });
    expect(w.step).toBe(1);
  });

  it('reset restores defaults', () => {
    const w = new Wizard();
    fillAccount(w);
    w.next();
    w.reset();
    expect(w.step).toBe(0);
    expect(w.data.email).toBe('');
    expect(w.verified).toEqual([]);
  });
});
