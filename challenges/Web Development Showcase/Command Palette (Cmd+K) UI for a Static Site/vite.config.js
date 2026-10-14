import { readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const root = fileURLToPath(new URL('.', import.meta.url));

// Every top-level .html file is a page of the demo site.
const pages = Object.fromEntries(
    readdirSync(root)
        .filter((name) => name.endsWith('.html'))
        .map((name) => [name.slice(0, -5), fileURLToPath(new URL(name, import.meta.url))]),
);

export default defineConfig({
    base: './',
    build: { rollupOptions: { input: pages } },
});
