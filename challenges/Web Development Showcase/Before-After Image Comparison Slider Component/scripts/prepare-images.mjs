// Builds the demo's before/after pairs from the Unsplash originals in raw/ (long edge <= 800px,
// WebP). Each "before" is the same photo with one visible defect added, so the pairs are honest
// about what they show: they are derived edits, not found before/after photographs.
import { mkdir } from 'node:fs/promises';
import sharp from 'sharp';

const OUT = 'demo/img';
await mkdir(OUT, { recursive: true });

const base = (name) =>
  sharp(`raw/${name}.jpg`).resize({ width: 800, height: 800, fit: 'inside', withoutEnlargement: true });

const pairs = [
  // Shadows lifted: the dark original vs a brightened, gamma-corrected edit.
  ['lightning', 'lightning-over-dark-mountain-peaks', (s) => s, (s) => s.gamma(1.8).modulate({ brightness: 1.45, saturation: 1.25 })],
  // Colour: grayscale vs the original.
  ['butterfly', 'painted-lady-butterfly-on-purple-salvia', (s) => s.grayscale(), (s) => s],
  // Resolution: a 96px-wide copy (shown upscaled) vs the 800px original.
  ['falls', 'whangarei-falls-footbridge', (s) => s.resize({ width: 96 }), (s) => s],
  // Portrait, used vertically: a dim, washed-out edit vs the original.
  ['stack', 'sea-stack-at-shark-fin-cove', (s) => s.modulate({ brightness: 0.6, saturation: 0.35 }), (s) => s],
];

for (const [id, file, before, after] of pairs) {
  for (const [side, edit] of [['before', before], ['after', after]]) {
    const info = await edit(base(file)).webp({ quality: 72 }).toFile(`${OUT}/${id}-${side}.webp`);
    console.log(`${id}-${side}\t${info.width}x${info.height}\t${Math.round(info.size / 1024)}kB`);
  }
}
