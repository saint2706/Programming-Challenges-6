import { existsSync } from 'node:fs';
import { join } from 'node:path';
import sharp from 'sharp';
import { describe, expect, it } from 'vitest';
import { PHOTOS } from '../src/photos.js';

// Masonry spans come from the declared width/height, so a wrong pair would
// silently overlap or gap tiles. Check the data against the real files.
describe('photo data', () => {
  it('has unique ids and non-empty credit fields', () => {
    expect(new Set(PHOTOS.map((p) => p.id)).size).toBe(PHOTOS.length);
    for (const p of PHOTOS) {
      for (const field of ['title', 'photographer', 'alt', 'url']) expect(p[field]?.trim(), `${p.id}.${field}`).toBeTruthy();
      expect(p.url).toBe(`https://unsplash.com/photos/${p.id}`);
    }
  });

  it.each(PHOTOS.map((p) => [p.id, p]))('%s: declared size matches the file on disk', async (_id, p) => {
    const file = join('public', p.src);
    expect(existsSync(file), `${file} missing`).toBe(true);
    const { width, height } = await sharp(file).metadata();
    expect({ width: p.width, height: p.height }).toEqual({ width, height });
  });
});
