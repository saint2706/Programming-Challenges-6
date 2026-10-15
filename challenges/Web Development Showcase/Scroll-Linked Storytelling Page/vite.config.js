import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { chartMarkup, dataTableMarkup } from './src/chart.js';
import { stripesMarkup } from './src/stripes.js';

const { years } = JSON.parse(
    readFileSync(fileURLToPath(new URL('./src/data/gistemp-annual.json', import.meta.url)), 'utf8'),
);

// The chart and the stripes are written into the HTML at build time, so they show without JavaScript.
export default defineConfig({
    base: './',
    plugins: [
        {
            name: 'static-figures',
            transformIndexHtml: (html) =>
                html
                    .replace('<!--chart-svg-->', chartMarkup(years))
                    .replace('<!--data-table-->', dataTableMarkup(years))
                    .replace('<!--stripes-->', stripesMarkup(years)),
        },
    ],
});
