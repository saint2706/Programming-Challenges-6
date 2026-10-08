import { afterEach, describe, expect, it, vi } from 'vitest';
import { emptyResume } from '../src/lib/schema.js';
import { seedResume } from '../src/lib/seed.js';
import {
  CORRUPT_KEY,
  STORAGE_KEY,
  STORAGE_VERSION,
  createAutosaver,
  createStorage,
} from '../src/lib/storage.js';

function memoryBackend(initial = {}) {
  const data = new Map(Object.entries(initial));
  return {
    data,
    getItem: (k) => (data.has(k) ? data.get(k) : null),
    setItem: (k, v) => void data.set(k, String(v)),
    removeItem: (k) => void data.delete(k),
  };
}

const throwing = () => {
  const boom = () => {
    throw new Error('SecurityError: storage is disabled');
  };
  return { getItem: boom, setItem: boom, removeItem: boom };
};

describe('createStorage', () => {
  it('reports an empty store', () => {
    expect(createStorage(memoryBackend()).load()).toEqual({ status: 'empty' });
  });

  it('round-trips a document with a version and timestamp', () => {
    const backend = memoryBackend();
    const storage = createStorage(backend);
    const doc = seedResume();
    expect(storage.save(doc)).toBe(true);
    const saved = JSON.parse(backend.data.get(STORAGE_KEY));
    expect(saved.version).toBe(STORAGE_VERSION);
    expect(new Date(saved.savedAt).toString()).not.toBe('Invalid Date');
    const loaded = storage.load();
    expect(loaded.status).toBe('ok');
    expect(loaded.doc).toEqual(doc);
  });

  it('keeps a corrupt value under the backup key instead of throwing it away', () => {
    const backend = memoryBackend({ [STORAGE_KEY]: '{not json' });
    const loaded = createStorage(backend).load();
    expect(loaded.status).toBe('corrupt');
    expect(backend.data.get(CORRUPT_KEY)).toBe('{not json');
  });

  it('treats a wrong version or an invalid document as corrupt', () => {
    const wrongVersion = memoryBackend({ [STORAGE_KEY]: JSON.stringify({ version: 99, doc: emptyResume() }) });
    expect(createStorage(wrongVersion).load().status).toBe('corrupt');

    const invalid = memoryBackend({
      [STORAGE_KEY]: JSON.stringify({ version: STORAGE_VERSION, doc: { work: [{ url: 'javascript:alert(1)' }] } }),
    });
    expect(createStorage(invalid).load().status).toBe('corrupt');

    const notObject = memoryBackend({ [STORAGE_KEY]: '42' });
    expect(createStorage(notObject).load().status).toBe('corrupt');
  });

  it('does not overwrite the backup with later good saves', () => {
    const backend = memoryBackend({ [STORAGE_KEY]: '{bad' });
    const storage = createStorage(backend);
    storage.load();
    storage.save(emptyResume());
    expect(backend.data.get(CORRUPT_KEY)).toBe('{bad');
    expect(createStorage(backend).load().status).toBe('ok');
  });

  it('works with storage disabled: nothing throws and nothing is saved', () => {
    const storage = createStorage(throwing());
    expect(storage.available).toBe(false);
    expect(storage.load()).toEqual({ status: 'empty' });
    expect(storage.save(emptyResume())).toBe(false);
    expect(() => storage.clear()).not.toThrow();
  });

  it('works when there is no storage object at all', () => {
    const storage = createStorage(null);
    expect(storage.load()).toEqual({ status: 'empty' });
    expect(storage.save(emptyResume())).toBe(false);
  });

  it('reports a failed save (quota exceeded) without throwing', () => {
    const backend = memoryBackend();
    backend.setItem = () => {
      throw new DOMException('quota', 'QuotaExceededError');
    };
    const storage = createStorage(backend);
    expect(storage.save(emptyResume())).toBe(false);
  });

  it('clears the saved document', () => {
    const backend = memoryBackend();
    const storage = createStorage(backend);
    storage.save(emptyResume());
    storage.clear();
    expect(storage.load()).toEqual({ status: 'empty' });
  });
});

describe('createAutosaver', () => {
  afterEach(() => vi.useRealTimers());

  it('saves only the latest document once things go quiet', () => {
    vi.useFakeTimers();
    const save = vi.fn();
    const saver = createAutosaver(save, 400);
    saver.schedule('a');
    vi.advanceTimersByTime(300);
    saver.schedule('b');
    vi.advanceTimersByTime(300);
    expect(save).not.toHaveBeenCalled();
    vi.advanceTimersByTime(100);
    expect(save).toHaveBeenCalledTimes(1);
    expect(save).toHaveBeenCalledWith('b');
  });

  it('flush saves immediately and cancels the timer', () => {
    vi.useFakeTimers();
    const save = vi.fn();
    const saver = createAutosaver(save, 400);
    saver.schedule('a');
    saver.flush();
    expect(save).toHaveBeenCalledWith('a');
    vi.advanceTimersByTime(1000);
    expect(save).toHaveBeenCalledTimes(1);
  });

  it('flush with nothing pending does nothing, and cancel drops the pending save', () => {
    vi.useFakeTimers();
    const save = vi.fn();
    const saver = createAutosaver(save, 400);
    saver.flush();
    saver.schedule('a');
    saver.cancel();
    vi.advanceTimersByTime(1000);
    expect(save).not.toHaveBeenCalled();
  });
});
