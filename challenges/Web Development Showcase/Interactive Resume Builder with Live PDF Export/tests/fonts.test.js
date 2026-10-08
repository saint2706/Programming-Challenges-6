import { describe, expect, it, vi } from 'vitest';
import { FONT_PAIRS, ensureFonts, fontFamilies, fontVars, onFontsLoaded } from '../src/lib/fonts.js';

function fakeFontSet({ failFor } = {}) {
  const listeners = new Map();
  return {
    loads: [],
    load(spec, text) {
      this.loads.push([spec, text]);
      return failFor && spec.includes(failFor) ? Promise.reject(new Error('nope')) : Promise.resolve([]);
    },
    ready: Promise.resolve(),
    addEventListener: vi.fn((type, fn) => listeners.set(type, fn)),
    removeEventListener: vi.fn((type) => listeners.delete(type)),
    emit: (type) => listeners.get(type)?.(),
  };
}

describe('font pairs', () => {
  it('defines inter, serif and plex', () => {
    expect(Object.keys(FONT_PAIRS)).toEqual(['inter', 'serif', 'plex']);
  });

  it('lists the distinct families a pair needs', () => {
    expect(fontFamilies('inter')).toEqual(['Inter']);
    expect(fontFamilies('serif')).toEqual(['Source Serif 4', 'Inter']);
    expect(fontFamilies('plex')).toEqual(['IBM Plex Sans']);
    expect(fontFamilies('nope')).toEqual(['Inter']);
  });

  it('turns a pair into CSS custom properties with fallbacks', () => {
    expect(fontVars('serif')['--font-heading']).toBe('"Source Serif 4", Georgia, serif');
    expect(fontVars('serif')['--font-body']).toBe('"Inter", system-ui, sans-serif');
  });
});

describe('ensureFonts', () => {
  it('loads each family at the four weights for the text on the page', async () => {
    const fonts = fakeFontSet();
    await ensureFonts('serif', 'Hello', fonts);
    expect(fonts.loads).toHaveLength(8);
    expect(fonts.loads[0]).toEqual(['400 16px "Source Serif 4"', 'Hello']);
    expect(fonts.loads.map(([spec]) => spec)).toContain('700 16px "Inter"');
    expect(fonts.loads.map(([spec]) => spec)).toContain('500 16px "Inter"');
  });

  it('does not throw when a font fails to load', async () => {
    const fonts = fakeFontSet({ failFor: 'Inter' });
    await expect(ensureFonts('inter', 'x', fonts)).resolves.toBeUndefined();
  });

  it('resolves immediately when there is no FontFaceSet', async () => {
    await expect(ensureFonts('inter', 'x', null)).resolves.toBeUndefined();
  });

  it('caps the sample text it passes to the font loader', async () => {
    const fonts = fakeFontSet();
    await ensureFonts('inter', 'a'.repeat(50_000), fonts);
    expect(fonts.loads[0][1].length).toBeLessThanOrEqual(4000);
  });
});

describe('onFontsLoaded', () => {
  it('calls back on loadingdone and can unsubscribe', () => {
    const fonts = fakeFontSet();
    const callback = vi.fn();
    const stop = onFontsLoaded(callback, fonts);
    fonts.emit('loadingdone');
    expect(callback).toHaveBeenCalledTimes(1);
    stop();
    expect(fonts.removeEventListener).toHaveBeenCalledWith('loadingdone', expect.any(Function));
    fonts.emit('loadingdone');
    expect(callback).toHaveBeenCalledTimes(1);
  });

  it('is a no-op without a FontFaceSet', () => {
    expect(onFontsLoaded(() => {}, null)).toBeTypeOf('function');
  });
});
