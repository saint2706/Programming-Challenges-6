/** `pnpm run dev`: the Hono server (API only, restarts on change) and Vite side by side. */
import { type ChildProcess, spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';

const children: ChildProcess[] = [];
const run = (args: string[], env: Record<string, string> = {}) => {
    // Plain `node` for both, so there is no shell involved and it behaves the same on every OS.
    const child = spawn(process.execPath, args, { stdio: 'inherit', env: { ...process.env, ...env } });
    child.on('exit', () => stop());
    children.push(child);
};

let stopping = false;
function stop(): void {
    if (stopping) return;
    stopping = true;
    for (const c of children) c.kill();
    process.exit(0);
}
process.on('SIGINT', stop);
process.on('SIGTERM', stop);

const vite = join(dirname(createRequire(import.meta.url).resolve('vite/package.json')), 'bin', 'vite.js');
run(['--watch', 'server/index.ts'], { WHITEBOARD_API_ONLY: '1', PORT: '8787' });
run([vite]);
