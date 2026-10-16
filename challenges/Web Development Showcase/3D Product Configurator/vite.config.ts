import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { type Plugin, defineConfig } from 'vitest/config';

const require = createRequire(import.meta.url);
const WASM = ['vision_wasm_internal.js', 'vision_wasm_internal.wasm'];
const TYPES: Record<string, string> = { js: 'text/javascript', wasm: 'application/wasm' };

/**
 * MediaPipe loads its WebAssembly runtime by URL at run time, so it cannot be bundled. Serve the
 * two files it needs (the SIMD build: every browser since 2023 has SIMD) from this site as
 * `mediapipe/`, in development and in the build, rather than from a CDN: nothing leaves the page.
 */
function mediapipe(): Plugin {
    const read = (name: string) => readFileSync(require.resolve(`@mediapipe/tasks-vision/${name}`));
    return {
        name: 'mediapipe-runtime',
        configureServer(server) {
            server.middlewares.use('/mediapipe/', (req, res, next) => {
                let name = '';
                try {
                    name = decodeURIComponent((req.url ?? '').split('?')[0]!.slice(1));
                } catch {
                    return next(); // not a file name
                }
                if (!WASM.includes(name)) return next();
                res.setHeader('Content-Type', TYPES[name.split('.').pop()!]!);
                res.end(read(name));
            });
        },
        generateBundle() {
            for (const name of WASM) this.emitFile({ type: 'asset', fileName: `mediapipe/${name}`, source: read(name) });
        },
    };
}

/** The page's content security policy asks for no inline styles; the development server injects some, so it is loosened there only. */
function devStyles(): Plugin {
    return { name: 'dev-inline-styles', apply: 'serve', transformIndexHtml: (html) => html.replace("style-src 'self'", "style-src 'self' 'unsafe-inline'") };
}

export default defineConfig({
    // Relative, so the app works from any folder, such as GitHub Pages.
    base: './',
    plugins: [svelte(), mediapipe(), devStyles()],
    build: { target: 'es2023' },
    test: {
        environment: 'node',
        include: ['tests/**/*.test.ts'],
    },
});
