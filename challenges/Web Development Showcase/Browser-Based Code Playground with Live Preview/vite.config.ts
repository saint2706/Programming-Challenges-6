import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { type Plugin } from 'vite';
import { defineConfig } from 'vitest/config';

const require = createRequire(import.meta.url);

/**
 * The packages the playground carries itself, as source text keyed by the name a person imports.
 * They are bundled into each run with the person's code (one copy, so hooks and signals share one
 * Preact), which means the preview needs no import map and works offline.
 */
function vendorSources(): Plugin {
    const id = 'virtual:vendor';
    const files: Record<string, string> = {
        preact: join(dirname(require.resolve('preact/package.json')), 'dist', 'preact.mjs'),
        'preact/hooks': join(dirname(require.resolve('preact/package.json')), 'hooks', 'dist', 'hooks.mjs'),
        'preact/jsx-runtime': join(dirname(require.resolve('preact/package.json')), 'jsx-runtime', 'dist', 'jsxRuntime.mjs'),
        'preact/jsx-dev-runtime': join(dirname(require.resolve('preact/package.json')), 'jsx-runtime', 'dist', 'jsxRuntime.mjs'),
        '@preact/signals': join(dirname(require.resolve('@preact/signals')), 'signals.mjs'),
        '@preact/signals-core': join(dirname(require.resolve('@preact/signals-core')), 'signals-core.mjs'),
    };
    return {
        name: 'vendor-sources',
        resolveId: (source) => (source === id ? `\0${id}` : null),
        load(loaded) {
            if (loaded !== `\0${id}`) return null;
            const sources: Record<string, string> = {};
            for (const [spec, path] of Object.entries(files)) {
                this.addWatchFile(path);
                sources[spec] = readFileSync(path, 'utf8');
            }
            return `export default ${JSON.stringify(sources)};`;
        },
    };
}

export default defineConfig({
    // Relative, so the app (and its sandbox pages) work from any folder, such as GitHub Pages.
    base: './',
    plugins: [vendorSources()],
    build: { target: 'es2023' },
    test: {
        environment: 'node',
        include: ['tests/**/*.test.ts'],
    },
});
