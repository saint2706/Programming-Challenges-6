/**
 * Coalesces repagination requests: any number of `request()` calls within one frame run the job
 * once, on the next animation frame. `flushNow()` runs a pending job synchronously (used before
 * printing, so the page is never printed stale).
 */

const browserFrames = {
  request: (fn) => globalThis.requestAnimationFrame(fn),
  cancel: (id) => globalThis.cancelAnimationFrame(id),
};

export function createFrameScheduler(run, frames = browserFrames) {
  let handle = null;

  return {
    get pending() {
      return handle !== null;
    },
    request() {
      if (handle !== null) return;
      handle = frames.request(() => {
        handle = null;
        run();
      });
    },
    flushNow() {
      if (handle === null) return;
      frames.cancel(handle);
      handle = null;
      run();
    },
    cancel() {
      if (handle === null) return;
      frames.cancel(handle);
      handle = null;
    },
  };
}
