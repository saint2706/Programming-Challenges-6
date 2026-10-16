import { defineConfig } from 'vitest/config';

// Real-browser tests: they drive an installed Chrome or Edge against the built app (dist/).
export default defineConfig({
    test: {
        environment: 'node',
        include: ['e2e/**/*.e2e.test.ts'],
        testTimeout: 90_000,
        hookTimeout: 90_000,
        fileParallelism: false,
    },
});
