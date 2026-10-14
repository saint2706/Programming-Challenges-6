import { defineConfig } from 'tsup';

export default defineConfig({
  entry: { index: 'src/index.js' },
  format: ['esm', 'cjs', 'iife'],
  globalName: 'ImgCompare',
  target: 'es2022',
  dts: true,
  sourcemap: true,
  clean: true,
  minify: true,
});
