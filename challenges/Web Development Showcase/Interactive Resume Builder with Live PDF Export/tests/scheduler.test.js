import { describe, expect, it, vi } from 'vitest';
import { createFrameScheduler } from '../src/lib/scheduler.js';

function fakeRaf() {
  const queue = new Map();
  let next = 1;
  return {
    request: (fn) => {
      queue.set(next, fn);
      return next++;
    },
    cancel: (id) => void queue.delete(id),
    tick: () => {
      const jobs = [...queue.values()];
      queue.clear();
      jobs.forEach((fn) => fn());
    },
    size: () => queue.size,
  };
}

describe('createFrameScheduler', () => {
  it('coalesces a burst of requests into one run per frame', () => {
    const raf = fakeRaf();
    const run = vi.fn();
    const scheduler = createFrameScheduler(run, raf);
    for (let i = 0; i < 100; i++) scheduler.request();
    expect(raf.size()).toBe(1);
    expect(run).not.toHaveBeenCalled();
    raf.tick();
    expect(run).toHaveBeenCalledTimes(1);
    expect(scheduler.pending).toBe(false);
  });

  it('schedules again after a run', () => {
    const raf = fakeRaf();
    const run = vi.fn();
    const scheduler = createFrameScheduler(run, raf);
    scheduler.request();
    raf.tick();
    scheduler.request();
    raf.tick();
    expect(run).toHaveBeenCalledTimes(2);
  });

  it('flushNow runs synchronously, once, and cancels the pending frame', () => {
    const raf = fakeRaf();
    const run = vi.fn();
    const scheduler = createFrameScheduler(run, raf);
    scheduler.request();
    scheduler.flushNow();
    expect(run).toHaveBeenCalledTimes(1);
    expect(raf.size()).toBe(0);
    raf.tick();
    expect(run).toHaveBeenCalledTimes(1);
  });

  it('flushNow with nothing pending does not run', () => {
    const run = vi.fn();
    createFrameScheduler(run, fakeRaf()).flushNow();
    expect(run).not.toHaveBeenCalled();
  });

  it('keeps working after the job throws', () => {
    const raf = fakeRaf();
    let fail = true;
    const run = vi.fn(() => {
      if (fail) throw new Error('boom');
    });
    const scheduler = createFrameScheduler(run, raf);
    scheduler.request();
    expect(() => raf.tick()).toThrow('boom');
    expect(scheduler.pending).toBe(false);
    fail = false;
    scheduler.request();
    raf.tick();
    expect(run).toHaveBeenCalledTimes(2);
  });

  it('cancel drops the pending run', () => {
    const raf = fakeRaf();
    const run = vi.fn();
    const scheduler = createFrameScheduler(run, raf);
    scheduler.request();
    scheduler.cancel();
    raf.tick();
    expect(run).not.toHaveBeenCalled();
  });
});
