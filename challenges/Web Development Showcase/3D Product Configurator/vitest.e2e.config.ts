import { defineConfig } from 'vitest/config';

// Real-browser tests: they drive an installed Chrome or Edge against the built app (dist/).
export default defineConfig({
    test: {
        environment: 'node',
        include: ['e2e/**/*.e2e.test.ts'],
        // Software rendering on a CI runner takes up to a minute over some of these.
        testTimeout: process.env.CI ? 150_000 : 90_000,
        hookTimeout: 90_000,
        fileParallelism: false,
        // On CI, list each test as it finishes, and stop early: a hung browser would otherwise cost the whole job's time with nothing to read.
        reporters: process.env.CI ? ['verbose'] : ['default'],
        bail: process.env.CI ? 4 : 0,
    },
});
