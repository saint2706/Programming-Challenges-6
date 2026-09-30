import { defaultData, SENSITIVE_FIELDS } from './schemas.js';
import { STEP_COUNT } from './navigation.js';

export const STORAGE_KEY = 'wizard-draft:v1';
const VERSION = 1;

/** Copy of `data` with every sensitive field removed. */
export function stripSensitive(data) {
  const out = { ...data };
  for (const key of SENSITIVE_FIELDS) delete out[key];
  return out;
}

export function saveDraft(storage, { data, step, verified }) {
  try {
    storage.setItem(STORAGE_KEY, JSON.stringify({ v: VERSION, step, verified, data: stripSensitive(data) }));
    return true;
  } catch {
    return false; // quota exceeded / storage disabled: the wizard still works, just not resumable
  }
}

export function clearDraft(storage) {
  try {
    storage.removeItem(STORAGE_KEY);
  } catch {
    /* nothing to do */
  }
}

/**
 * Load a draft, trusting nothing: unknown keys are dropped, values whose type differs
 * from the default are ignored, sensitive fields are never read back. Returns null when
 * nothing usable is stored.
 */
export function loadDraft(storage) {
  let parsed;
  try {
    const raw = storage.getItem(STORAGE_KEY);
    if (!raw) return null;
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!parsed || typeof parsed !== 'object' || parsed.v !== VERSION) return null;

  const base = defaultData();
  const data = { ...base };
  const stored = parsed.data && typeof parsed.data === 'object' ? parsed.data : {};
  for (const key of Object.keys(base)) {
    if (SENSITIVE_FIELDS.includes(key) || !(key in stored)) continue;
    const value = stored[key];
    if (Array.isArray(base[key])) {
      if (Array.isArray(value) && value.every((x) => typeof x === 'string')) data[key] = value;
    } else if (typeof value === typeof base[key]) {
      data[key] = value;
    }
  }

  const verified = Array.isArray(parsed.verified)
    ? [...new Set(parsed.verified.filter((i) => Number.isInteger(i) && i >= 0 && i < STEP_COUNT))]
    : [];
  const step = Number.isInteger(parsed.step) && parsed.step >= 0 && parsed.step < STEP_COUNT ? parsed.step : 0;
  return { data, step, verified };
}
