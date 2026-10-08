import { describe, expect, it } from 'vitest';
import { announce, announcer } from '../src/lib/announcer.svelte.js';

describe('announce', () => {
  it('publishes a message on the next microtask', async () => {
    announce('Moved Experience up');
    expect(announcer.message).toBe('');
    await Promise.resolve();
    expect(announcer.message).toBe('Moved Experience up');
  });

  it('re-announces an identical message by clearing it first', async () => {
    announce('Done');
    await Promise.resolve();
    const seen = [announcer.message];
    announce('Done');
    seen.push(announcer.message);
    await Promise.resolve();
    seen.push(announcer.message);
    expect(seen).toEqual(['Done', '', 'Done']);
  });
});
