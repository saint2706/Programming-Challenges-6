// @vitest-environment node
import { describe, expect, it } from 'vitest';

describe('importing without a DOM (server-side rendering)', () => {
  it('does not throw, and define() reports that nothing was registered', async () => {
    expect(typeof HTMLElement).toBe('undefined');
    expect(typeof customElements).toBe('undefined');
    const mod = await import('../src/index.js');
    expect(mod.define()).toBe(false);
    expect(typeof mod.ImgCompare).toBe('function');
    expect(mod.keyTarget('End', 10)).toBe(100); // the pure helpers still work on the server
  });
});
