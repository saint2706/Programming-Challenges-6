import { Hono } from 'hono';
import { serveStatic } from '@hono/node-server/serve-static';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { appendRsvp, readRsvps } from './store.js';
import { validateRsvp } from './validate.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC_DIR = path.join(__dirname, '..', 'public');

/**
 * Builds the Hono app. This function IS the "serverless handler": Hono
 * apps expose `app.fetch(request)`, the same Request-in/Response-out
 * contract a Vercel Edge Function, a Cloudflare Worker, or an AWS Lambda
 * (via a Hono Lambda adapter) expects. `dev-server.js` adapts it to a
 * local Node socket for `npm run dev`; a real deployment would point a
 * platform's function runtime at `app.fetch` directly instead - no
 * handler code changes needed, only swapping `store.js`'s JSON file for
 * that platform's KV/database binding.
 */
export function createApp({ dataFile } = {}) {
  const app = new Hono();

  app.get('/api/rsvps', async (c) => {
    const rsvps = await readRsvps(dataFile);
    return c.json(rsvps);
  });

  app.post('/api/rsvps', async (c) => {
    let body;
    try {
      body = await c.req.json();
    } catch {
      return c.json({ error: 'Request body must be valid JSON' }, 400);
    }

    const result = validateRsvp(body);
    if (!result.ok) {
      return c.json({ error: result.error }, 400);
    }

    const rsvp = { ...result.value, submittedAt: new Date().toISOString() };
    await appendRsvp(rsvp, dataFile);
    return c.json(rsvp, 201);
  });

  // Registered after the API routes so it only ever handles what they don't.
  app.use('/*', serveStatic({ root: PUBLIC_DIR }));

  return app;
}
