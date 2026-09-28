import { defineConfig } from 'tsup';

export default defineConfig({
  entry: { index: 'src/index.js' },
  format: ['esm', 'cjs', 'iife'],
  globalName: 'TypingHero',
  dts: true,
  sourcemap: true,
  clean: true,
  minify: true,
});
