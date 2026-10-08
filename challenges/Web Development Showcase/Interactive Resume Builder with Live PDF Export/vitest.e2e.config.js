import { defineConfig } from 'vitest/config';

// The real-browser tests: they drive an installed Chrome or Edge against the built app (`dist/`).
export default defineConfig({
  test: {
    environment: 'node',
    include: ['e2e/**/*.e2e.test.js'],
    testTimeout: 120_000,
    hookTimeout: 120_000,
    fileParallelism: false,
  },
});
