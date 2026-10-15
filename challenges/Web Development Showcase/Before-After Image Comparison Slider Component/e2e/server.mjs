import { createReadStream, existsSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = normalize(fileURLToPath(new URL('..', import.meta.url))).replace(/[\\/]$/, '');
const MAX_DELAY_MS = 10_000;
const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
};

/**
 * Serves the package directory (so `/dist/index.js`, `/demo/`, `/e2e/harness.html` work) plus
 * `/slow.svg?ms=N`, an image that takes N ms to arrive. `requests` records every path asked for.
 */
export async function startServer(port = 0) {
  const requests = [];
  const server = createServer((req, res) => {
    const url = new URL(req.url, 'http://x');
    requests.push(url.pathname + url.search);
    if (url.pathname === '/slow.svg') {
      // The delay comes from the query string, so reject anything that would set a huge timer.
      const ms = Number(url.searchParams.get('ms') ?? 300);
      if (!(ms >= 0 && ms <= MAX_DELAY_MS)) {
        res.writeHead(400).end('ms must be between 0 and ' + MAX_DELAY_MS);
        return;
      }
      const [w, h] = [Number(url.searchParams.get('w') ?? 400), Number(url.searchParams.get('h') ?? 200)];
      setTimeout(() => {
        res.writeHead(200, { 'content-type': TYPES['.svg'] });
        res.end(`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}"><rect width="100%" height="100%" fill="#2a9d8f"/></svg>`);
      }, ms);
      return;
    }
    const file = normalize(join(ROOT, decodeURIComponent(url.pathname)));
    if (!file.startsWith(ROOT + sep) || !existsSync(file) || !statSync(file).isFile()) {
      res.writeHead(404).end('not found');
      return;
    }
    res.writeHead(200, { 'content-type': TYPES[extname(file)] ?? 'application/octet-stream' });
    createReadStream(file).pipe(res);
  });
  await new Promise((resolve) => server.listen(port, '127.0.0.1', resolve));
  const url = `http://127.0.0.1:${server.address().port}`;
  return { url, requests, close: () => new Promise((resolve) => server.close(resolve)) };
}
