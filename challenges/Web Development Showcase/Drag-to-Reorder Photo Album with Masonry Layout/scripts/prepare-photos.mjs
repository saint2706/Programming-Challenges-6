// Downsizes the originals in raw/ (long edge <= 800px) to WebP in public/photos/
// and prints each file's dimensions for src/photos.js.
import { readdir, mkdir } from 'node:fs/promises';
import { basename, extname } from 'node:path';
import sharp from 'sharp';

const LONG_EDGE = 800;
await mkdir('public/photos', { recursive: true });
for (const file of (await readdir('raw')).filter((f) => f.endsWith('.jpg')).sort()) {
  const name = basename(file, extname(file));
  const info = await sharp(`raw/${file}`)
    .resize({ width: LONG_EDGE, height: LONG_EDGE, fit: 'inside', withoutEnlargement: true })
    .webp({ quality: 72 })
    .toFile(`public/photos/${name}.webp`);
  console.log(`${name}\t${info.width}x${info.height}\t${Math.round(info.size / 1024)}kB`);
}
