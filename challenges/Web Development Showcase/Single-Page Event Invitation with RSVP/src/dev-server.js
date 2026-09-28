import { serve } from '@hono/node-server';
import { createApp } from './server.js';

const app = createApp();
const port = process.env.PORT ? Number(process.env.PORT) : 3000;

serve({ fetch: app.fetch, port }, (info) => {
  console.log(`Event invitation RSVP server: http://localhost:${info.port}`);
});
