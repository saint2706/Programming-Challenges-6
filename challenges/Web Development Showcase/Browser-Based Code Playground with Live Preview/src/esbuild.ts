import type * as Esbuild from 'esbuild-wasm';

let ready: Promise<typeof Esbuild> | null = null;

/**
 * esbuild, compiled to WebAssembly (about 11 MB), loaded the first time a run needs it and then kept.
 * It runs in a worker, so building never blocks typing. A failed load is not remembered: the next run tries again.
 */
export function loadEsbuild(): Promise<typeof Esbuild> {
    ready ??= (async () => {
        const [esbuild, wasm] = await Promise.all([import('esbuild-wasm'), import('esbuild-wasm/esbuild.wasm?url')]);
        await esbuild.initialize({ wasmURL: wasm.default, worker: true });
        return esbuild;
    })().catch((error) => {
        ready = null;
        throw error;
    });
    return ready;
}
