import { defineConfig } from 'vitest/config';

// Real-browser tests: they drive an installed Chrome or Edge against the built app (dist/),
// served by the real Hono server.
export default defineConfig({
    test: {
        environment: 'node',
        include: ['e2e/**/*.e2e.test.ts'],
        testTimeout: 60_000,
        hookTimeout: 60_000,
        fileParallelism: false,
    },
});
