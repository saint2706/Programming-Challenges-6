import { defineConfig } from 'vitest/config';

export default defineConfig({
  base: './',
  test: {
    environment: 'node', // DOM tests opt in with `// @vitest-environment jsdom`
    include: ['tests/**/*.test.js'],
  },
});
