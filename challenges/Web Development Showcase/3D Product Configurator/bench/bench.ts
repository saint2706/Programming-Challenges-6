// `pnpm run bench`: what loading, drawing and tracking cost, on this machine, in real Chrome against dist/.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { gzipSync } from 'node:zlib';
import { fileURLToPath } from 'node:url';
import { cameraAbove, defaults, screenFromDiagonal } from '../src/headcoupled/calibration.ts';
import { HeadPipeline } from '../src/headcoupled/pipeline.ts';
import { CAMERA, faceLandmarks } from '../tests/helpers.ts';
import { launchChrome } from '../e2e/chrome.ts';
import { openStudio, read, startServer } from '../e2e/support.ts';

const dist = fileURLToPath(new URL('../dist/', import.meta.url));
const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)]!;
const kb = (n: number) => `${(n / 1024).toFixed(0)} kB`;

function sizes() {
    const rows: [string, number, number][] = [];
    const walk = (dir: string) => {
        for (const entry of readdirSync(dir, { withFileTypes: true })) {
            const path = dir + entry.name;
            if (entry.isDirectory()) walk(path + '/');
            else if (/\.(js|css|glb|hdr|task|wasm|png)$/.test(entry.name)) rows.push([path.slice(dist.length), statSync(path).size, /\.(js|css|wasm|hdr)$/.test(entry.name) ? gzipSync(readFileSync(path)).length : 0]);
        }
    };
    walk(dist);
    return rows.sort((a, b) => b[1] - a[1]);
}

console.log('| File | Size | Gzipped |\n| --- | ---: | ---: |');
for (const [name, size, gz] of sizes()) console.log(`| \`${name}\` | ${kb(size)} | ${gz ? kb(gz) : ''} |`);

const server = await startServer();
const plain = await launchChrome();
const withCamera = await launchChrome(['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream']);

// Cold start: a new profile each time, from navigation to the first picture of the shoe.
const starts: number[] = [];
let backend = '';
for (let i = 0; i < 5; i++) {
    const t0 = performance.now();
    const { page } = await openStudio(plain, server.url);
    starts.push(performance.now() - t0);
    backend = await read(page, 'studio.backend');
    await page.close();
}
console.log(`\ncold start to the first picture (${backend}): median ${median(starts).toFixed(0)} ms (${starts.map((s) => s.toFixed(0)).join(', ')})`);

// The window preview: how many frames a second, and how much of the main thread they take.
{
    const { page } = await openStudio(plain, server.url, { query: '&mode=window' });
    const client = await page.createCDPSession();
    await client.send('Performance.enable');
    const metric = async (name: string) => (await client.send('Performance.getMetrics')).metrics.find((m) => m.name === name)!.value;
    await new Promise((r) => setTimeout(r, 1500));
    const [task0, frames0, t0] = [await metric('TaskDuration'), await read<number>(page, 'studio.frames'), performance.now()];
    await new Promise((r) => setTimeout(r, 4000));
    const [task1, frames1, t1] = [await metric('TaskDuration'), await read<number>(page, 'studio.frames'), performance.now()];
    const seconds = (t1 - t0) / 1000;
    console.log(`window preview: ${((frames1 - frames0) / seconds).toFixed(0)} frames/s, main thread busy ${(((task1 - task0) / seconds) * 100).toFixed(0)}% (${(((task1 - task0) / (frames1 - frames0)) * 1000).toFixed(2)} ms per frame)`);
    await page.close();
}

// The face model on a webcam stream (a test pattern with no face: the model still looks every frame).
{
    const { page } = await openStudio(withCamera, server.url, { query: '&mode=window' });
    await page.click('.panel .btn.primary');
    await page.waitForFunction(() => ['no-face', 'tracking'].includes((window as unknown as { __app: { head: { status: string } } }).__app.head.status), { timeout: 90_000 });
    await new Promise((r) => setTimeout(r, 1500));
    const samples: number[] = [];
    for (let i = 0; i < 40; i++) {
        samples.push(await read<number>(page, 'head.inferenceMs'));
        await new Promise((r) => setTimeout(r, 100));
    }
    console.log(`face model (${await read(page, 'head.delegate')}, camera at ${await page.evaluate(() => { const v = (window as unknown as { __app: { head: { video: HTMLVideoElement } } }).__app.head.video; return `${v.videoWidth}x${v.videoHeight}`; })}): ${await read(page, 'head.fps')} frames/s, ${median(samples).toFixed(1)} ms per frame (median of 40 readings)`);
    await page.close();
}

// The maths of one frame: landmarks to an eye, in the pipeline.
{
    const s = defaults(1536, 864);
    const above = cameraAbove(screenFromDiagonal(s.diagonal, s.aspect));
    const lm = faceLandmarks({ x: -0.1, y: above - 0.02, z: 0.55 }, 0.063, CAMERA);
    const pipeline = new HeadPipeline(s);
    const n = 200_000;
    const t0 = performance.now();
    for (let i = 0; i < n; i++) pipeline.update(lm, CAMERA, i / 30);
    console.log(`one frame of head maths: ${(((performance.now() - t0) / n) * 1000).toFixed(2)} µs`);
}

await plain.close();
await withCamera.close();
await server.close();
