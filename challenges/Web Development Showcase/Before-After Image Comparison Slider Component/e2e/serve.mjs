// Serves the package so the demo (which imports ../dist/index.js as a module) works over http.
import { startServer } from './server.mjs';

const { url } = await startServer(Number(process.env.PORT ?? 4173));
console.log(`demo: ${url}/demo/index.html`);
