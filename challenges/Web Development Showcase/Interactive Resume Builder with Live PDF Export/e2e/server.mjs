import { fileURLToPath } from 'node:url';
import { preview } from 'vite';

/** Serve the built app (`dist/`) on a free local port. Run `vite build` first. */
export async function startServer() {
  const server = await preview({
    root: fileURLToPath(new URL('..', import.meta.url)),
    logLevel: 'silent',
    preview: { host: '127.0.0.1', port: 4173, strictPort: false },
  });
  const url = server.resolvedUrls.local[0].replace(/\/$/, '');
  return { url, close: () => server.close() };
}
