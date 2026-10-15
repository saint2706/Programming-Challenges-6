import { defineConfig } from 'vite';

const API = process.env.WHITEBOARD_API ?? 'http://127.0.0.1:8787';

export default defineConfig({
    build: { target: 'es2023' },
    server: {
        // In development the page comes from Vite and the WebSocket goes to the Hono server.
        proxy: { '/ws': { target: API, ws: true } },
    },
});
