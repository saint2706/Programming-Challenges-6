import { fileURLToPath } from 'node:url';
import { startServer } from './app.ts';

const root = fileURLToPath(new URL('..', import.meta.url));
const port = Number(process.env.PORT ?? 8787);

const running = await startServer({
    dataDir: process.env.WHITEBOARD_DATA_DIR ?? `${root}data/rooms`,
    // `pnpm run dev` serves the client from Vite and only needs the API here.
    staticDir: process.env.WHITEBOARD_API_ONLY ? undefined : `${root}dist`,
    port,
    hostname: process.env.HOST ?? '127.0.0.1',
    allowedOrigins: process.env.ALLOWED_ORIGINS?.split(',').filter(Boolean),
    allowedHosts: process.env.ALLOWED_HOSTS?.split(',').filter(Boolean),
});
console.log(`whiteboard listening on ${running.url}`);

for (const signal of ['SIGINT', 'SIGTERM'] as const) {
    process.once(signal, () => {
        void running.close().then(() => process.exit(0));
    });
}
