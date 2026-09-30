import { describe, it, expect } from 'vitest';
import { STORAGE_KEY, clearDraft, loadDraft, saveDraft, stripSensitive } from '../src/lib/storage.js';
import { defaultData } from '../src/lib/schemas.js';

function memoryStore(initial = {}) {
  const map = new Map(Object.entries(initial));
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => void map.set(k, String(v)),
    removeItem: (k) => void map.delete(k),
    raw: () => map.get(STORAGE_KEY),
  };
}

describe('saveDraft', () => {
  it('never writes passwords or card data', () => {
    const store = memoryStore();
    const data = {
      ...defaultData(),
      email: 'a@b.co',
      password: 'hunter2hunter2',
      confirmPassword: 'hunter2hunter2',
      cardNumber: '4242424242424242',
      cvc: '123',
      expiry: '12/30',
    };
    saveDraft(store, { data, step: 3, verified: [0, 1, 2] });
    const raw = store.raw();
    expect(raw).toContain('a@b.co');
    for (const secret of ['hunter2', '4242424242424242', '"cvc"', '12/30']) expect(raw).not.toContain(secret);
  });

  it('returns false instead of throwing when storage rejects the write', () => {
    const broken = {
      setItem: () => {
        throw new Error('QuotaExceededError');
      },
    };
    expect(saveDraft(broken, { data: defaultData(), step: 0, verified: [] })).toBe(false);
  });
});

describe('loadDraft', () => {
  it('round-trips a saved draft', () => {
    const store = memoryStore();
    const data = { ...defaultData(), fullName: 'Ada', interests: ['data', 'ml'], newsletter: true };
    saveDraft(store, { data, step: 2, verified: [0, 1] });
    expect(loadDraft(store)).toEqual({ data: { ...defaultData(), ...stripSensitive(data) }, step: 2, verified: [0, 1] });
  });

  it('returns null for empty, corrupt, or wrong-version storage', () => {
    expect(loadDraft(memoryStore())).toBeNull();
    expect(loadDraft(memoryStore({ [STORAGE_KEY]: '{not json' }))).toBeNull();
    expect(loadDraft(memoryStore({ [STORAGE_KEY]: JSON.stringify({ v: 99, data: {} }) }))).toBeNull();
    expect(loadDraft(memoryStore({ [STORAGE_KEY]: 'null' }))).toBeNull();
  });

  it('tolerates a throwing getItem', () => {
    expect(
      loadDraft({
        getItem: () => {
          throw new Error('SecurityError');
        },
      }),
    ).toBeNull();
  });

  it('drops unknown keys, wrongly-typed values, and out-of-range steps', () => {
    const store = memoryStore({
      [STORAGE_KEY]: JSON.stringify({
        v: 1,
        step: 42,
        verified: [0, 1, 1, 'x', 99, -2, 1.5],
        data: { email: 123, fullName: 'Grace', evil: '<script>', interests: ['a', 5], newsletter: 'yes' },
      }),
    });
    const draft = loadDraft(store);
    expect(draft.step).toBe(0);
    expect(draft.verified).toEqual([0, 1]);
    expect(draft.data.email).toBe('');
    expect(draft.data.fullName).toBe('Grace');
    expect(draft.data.interests).toEqual([]);
    expect(draft.data.newsletter).toBe(false);
    expect('evil' in draft.data).toBe(false);
  });

  it('ignores sensitive fields even if a tampered draft contains them', () => {
    const store = memoryStore({
      [STORAGE_KEY]: JSON.stringify({ v: 1, step: 0, verified: [], data: { password: 'x'.repeat(12), cardNumber: '4242424242424242' } }),
    });
    const draft = loadDraft(store);
    expect(draft.data.password).toBe('');
    expect(draft.data.cardNumber).toBe('');
  });
});

describe('clearDraft', () => {
  it('removes the entry', () => {
    const store = memoryStore();
    saveDraft(store, { data: defaultData(), step: 0, verified: [] });
    clearDraft(store);
    expect(store.getItem(STORAGE_KEY)).toBeNull();
  });
});
