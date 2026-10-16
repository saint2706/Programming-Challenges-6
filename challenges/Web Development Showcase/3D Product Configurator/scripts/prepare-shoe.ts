// Build-time only: turns Khronos's MaterialsVariantsShoe sample into the configurator's asset,
// public/models/shoe.glb. Run with `pnpm run assets` (needs network once; the output is committed).
//
// The sample is one mesh with three whole-texture variants (Beach, Midnight, Street), so it cannot
// be recoloured part by part as it is. Its UV islands, though, line up with the parts of a shoe
// (each island sits on one flat region of the colour texture), so:
//
//   1. split the mesh into one mesh per part, by island;
//   2. replace the three colour textures with one grey "detail" map (knit, stitching, shading) in
//      which every island is divided by its own average, so a part's colour is a single multiplier
//      at run time and the weave survives any colour;
//   3. keep the sample's normal and occlusion/roughness/metalness maps, shrink and compress all of
//      it (WebP textures, meshopt geometry);
//   4. draw the shoe's shadow on the floor (public/models/shadow.png): the sole's footprint, sharp,
//      under the whole shoe's, soft and weaker the higher a part is.
//
// Prints the average colour each part has in each of the three variants (the configurator's
// "Beach", "Midnight" and "Street" presets are those numbers).
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Document, NodeIO, type Texture } from '@gltf-transform/core';
import { ALL_EXTENSIONS, KHRMaterialsClearcoat } from '@gltf-transform/extensions';
import { meshopt, textureCompress } from '@gltf-transform/functions';
import { MeshoptDecoder, MeshoptEncoder } from 'meshoptimizer';
import sharp from 'sharp';
import { PART_IDS } from '../src/parts.ts';
import { SHADOW, SHOE } from '../src/shoe/dimensions.ts';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const SOURCE_URL = 'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/MaterialsVariantsShoe/glTF-Binary/MaterialsVariantsShoe.glb';
const SOURCE_SHA256 = 'e1d7cb190382111e5a5b37b51e9a7f007f7eb2ab1b6185e0188e8d0a0d1265a7';
const cache = resolve(root, '.cache/MaterialsVariantsShoe.glb');
const output = resolve(root, 'public/models/shoe.glb');
const shadowOutput = resolve(root, 'public/models/shadow.png');

/** Which connected pieces of the mesh are which part. Island numbers are by first triangle, so they are stable for this file. */
const RANGES: [part: string, from: number, to: number][] = [
    ['laces', 0, 6],
    ['upper', 7, 7],
    ['lining', 8, 16],
    ['upper', 17, 24],
    ['accents', 25, 48], // pull loops and eyelets
    ['midsole', 49, 52],
    ['outsole', 53, 53],
    ['midsole', 54, 57],
    ['overlays', 58, 82], // heel patch and the side straps
];
const PARTS = PART_IDS;
const ISLANDS = 83;

const srgbToLinear = Float32Array.from({ length: 256 }, (_, i) => {
    const c = i / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
});
const linearToSrgb = (x: number) => (x <= 0.0031308 ? x * 12.92 : 1.055 * x ** (1 / 2.4) - 0.055);

/** Nothing but the exact file this script was written for is ever read, and only a checked download is kept. */
function check(bytes: Uint8Array, what: string): void {
    const sha = createHash('sha256').update(bytes).digest('hex');
    if (sha !== SOURCE_SHA256) throw new Error(`${what} is not the file this script was written for (sha256 ${sha})`);
}

async function source(): Promise<Uint8Array> {
    if (!existsSync(cache)) {
        console.log('downloading', SOURCE_URL);
        const response = await fetch(SOURCE_URL);
        if (!response.ok) throw new Error(`download failed: ${response.status}`);
        const body = new Uint8Array(await response.arrayBuffer());
        if (body.byteLength > 50 * 1024 * 1024) throw new Error('the download is far bigger than the model it should be');
        check(body, 'the download');
        mkdirSync(dirname(cache), { recursive: true });
        writeFileSync(cache, body);
    }
    const bytes = readFileSync(cache);
    check(bytes, cache);
    return bytes;
}

/** Connected pieces of an indexed triangle mesh, numbered in order of their first triangle. */
function islandsOf(indices: ArrayLike<number>, vertexCount: number): { of: Int32Array; count: number } {
    const parent = Int32Array.from({ length: vertexCount }, (_, i) => i);
    const find = (a: number): number => {
        while (parent[a] !== a) {
            parent[a] = parent[parent[a]!]!;
            a = parent[a]!;
        }
        return a;
    };
    for (let i = 0; i < indices.length; i += 3) {
        const a = find(indices[i]!);
        parent[find(indices[i + 1]!)] = a;
        parent[find(indices[i + 2]!)] = a;
    }
    const number = new Map<number, number>();
    const of = new Int32Array(vertexCount).fill(-1);
    for (let i = 0; i < indices.length; i += 3) {
        const root = find(indices[i]!);
        if (!number.has(root)) number.set(root, number.size);
        for (let k = 0; k < 3; k++) of[indices[i + k]!] = number.get(root)!;
    }
    return { of, count: number.size };
}

interface Raster {
    owner: Int16Array;
    /** Texels actually under a triangle, before the borders were grown. */
    covered: Uint8Array;
}

/** Which island each texel of the colour texture belongs to, and then the same grown a few texels past the edges. */
function rasterize(uv: ArrayLike<number>, indices: ArrayLike<number>, islandOf: Int32Array, size: number): Raster {
    const owner = new Int16Array(size * size).fill(-1);
    for (let t = 0; t < indices.length; t += 3) {
        const a = indices[t]!;
        const b = indices[t + 1]!;
        const c = indices[t + 2]!;
        const ax = uv[a * 2]! * size;
        const ay = uv[a * 2 + 1]! * size;
        const bx = uv[b * 2]! * size;
        const by = uv[b * 2 + 1]! * size;
        const cx = uv[c * 2]! * size;
        const cy = uv[c * 2 + 1]! * size;
        const area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
        if (Math.abs(area) < 1e-9) continue;
        const x0 = Math.max(0, Math.floor(Math.min(ax, bx, cx)));
        const x1 = Math.min(size - 1, Math.ceil(Math.max(ax, bx, cx)));
        const y0 = Math.max(0, Math.floor(Math.min(ay, by, cy)));
        const y1 = Math.min(size - 1, Math.ceil(Math.max(ay, by, cy)));
        for (let y = y0; y <= y1; y++) {
            for (let x = x0; x <= x1; x++) {
                const px = x + 0.5;
                const py = y + 0.5;
                const w0 = ((bx - px) * (cy - py) - (cx - px) * (by - py)) / area;
                const w1 = ((cx - px) * (ay - py) - (ax - px) * (cy - py)) / area;
                const w2 = 1 - w0 - w1;
                if (w0 >= -0.01 && w1 >= -0.01 && w2 >= -0.01) owner[y * size + x] = islandOf[a]!;
            }
        }
    }
    const covered = Uint8Array.from(owner, (o) => (o >= 0 ? 1 : 0));
    // The artist's texture runs a few texels past each island so mip-mapping does not show seams: do the same.
    for (let pass = 0; pass < 8; pass++) {
        const next = owner.slice();
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const i = y * size + x;
                if (owner[i]! >= 0) continue;
                let n = -1;
                if (x > 0 && owner[i - 1]! >= 0) n = owner[i - 1]!;
                else if (x < size - 1 && owner[i + 1]! >= 0) n = owner[i + 1]!;
                else if (y > 0 && owner[i - size]! >= 0) n = owner[i - size]!;
                else if (y < size - 1 && owner[i + size]! >= 0) n = owner[i + size]!;
                if (n >= 0) next[i] = n;
            }
        }
        owner.set(next);
    }
    return { owner, covered };
}

/** The shadow picture: 0 is no shadow, 255 is full. Looks straight down at the floor (x across the picture, z down it). */
async function contactShadow(world: Float32Array, indices: ArrayLike<number>): Promise<Buffer> {
    const n = SHADOW.pixels;
    const contact = new Float32Array(n * n);
    const body = new Float32Array(n * n);
    for (let t = 0; t < indices.length; t += 3) {
        const v = [indices[t]!, indices[t + 1]!, indices[t + 2]!];
        const px = v.map((i) => (world[i * 3]! / SHADOW.width + 0.5) * n);
        const pz = v.map((i) => (world[i * 3 + 2]! / SHADOW.depth + 0.5) * n);
        const height = (world[v[0]! * 3 + 1]! + world[v[1]! * 3 + 1]! + world[v[2]! * 3 + 1]!) / 3;
        const area = (px[1]! - px[0]!) * (pz[2]! - pz[0]!) - (px[2]! - px[0]!) * (pz[1]! - pz[0]!);
        if (Math.abs(area) < 1e-9) continue;
        const weight = Math.max(0.25, 1 - height / 0.12);
        for (let y = Math.max(0, Math.floor(Math.min(...pz))); y <= Math.min(n - 1, Math.ceil(Math.max(...pz))); y++) {
            for (let x = Math.max(0, Math.floor(Math.min(...px))); x <= Math.min(n - 1, Math.ceil(Math.max(...px))); x++) {
                const qx = x + 0.5;
                const qy = y + 0.5;
                const w0 = ((px[1]! - qx) * (pz[2]! - qy) - (px[2]! - qx) * (pz[1]! - qy)) / area;
                const w1 = ((px[2]! - qx) * (pz[0]! - qy) - (px[0]! - qx) * (pz[2]! - qy)) / area;
                if (w0 < -0.01 || w1 < -0.01 || 1 - w0 - w1 < -0.01) continue;
                body[y * n + x] = Math.max(body[y * n + x]!, weight);
                if (height < 0.01) contact[y * n + x] = 1;
            }
        }
    }
    const blurred = async (mask: Float32Array, sigma: number) => sharp(Buffer.from(mask.map((m) => m * 255)), { raw: { width: n, height: n, channels: 1 } }).blur(sigma).extractChannel(0).raw().toBuffer();
    const [sharpEdge, softEdge] = await Promise.all([blurred(contact, 2.5), blurred(body, 9)]);
    const out = Buffer.alloc(n * n);
    for (let i = 0; i < n * n; i++) out[i] = Math.round(Math.max(sharpEdge[i]! * 0.9, softEdge[i]! * 0.6));
    return sharp(out, { raw: { width: n, height: n, channels: 1 } }).png({ compressionLevel: 9 }).toBuffer();
}

async function rgbOf(texture: Texture): Promise<{ data: Uint8Array; size: number }> {
    const { data, info } = await sharp(texture.getImage()!).removeAlpha().raw().toBuffer({ resolveWithObject: true });
    if (info.width !== info.height) throw new Error('expected a square texture');
    return { data, size: info.width };
}

const hex = (rgb: number[]) => `#${rgb.map((v) => Math.round(Math.min(1, Math.max(0, linearToSrgb(v))) * 255).toString(16).padStart(2, '0')).join('')}`;

async function main() {
    const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({ 'meshopt.decoder': MeshoptDecoder, 'meshopt.encoder': MeshoptEncoder });
    await MeshoptEncoder.ready;
    await MeshoptDecoder.ready;
    const src = await io.readBinary(await source());
    const node = src.getRoot().listNodes().find((n) => n.getMesh());
    const prim = node?.getMesh()?.listPrimitives()[0];
    if (!node || !prim) throw new Error('no mesh');
    const position = prim.getAttribute('POSITION')!.getArray()!;
    const normal = prim.getAttribute('NORMAL')!.getArray()!;
    const uv = prim.getAttribute('TEXCOORD_0')!.getArray()!;
    const indices = prim.getIndices()!.getArray()!;
    const vertexCount = position.length / 3;

    // To metres, then the shoe stands on the floor (y = 0) and is centred on the origin.
    const m = node.getWorldMatrix();
    const world = new Float32Array(position.length);
    for (let i = 0; i < vertexCount; i++) {
        const x = position[i * 3]!;
        const y = position[i * 3 + 1]!;
        const z = position[i * 3 + 2]!;
        for (let d = 0; d < 3; d++) world[i * 3 + d] = m[d]! * x + m[4 + d]! * y + m[8 + d]! * z + m[12 + d]!;
    }
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < vertexCount; i++) {
        for (let d = 0; d < 3; d++) {
            lo[d] = Math.min(lo[d]!, world[i * 3 + d]!);
            hi[d] = Math.max(hi[d]!, world[i * 3 + d]!);
        }
    }
    const shift = [(lo[0]! + hi[0]!) / 2, lo[1]!, (lo[2]! + hi[2]!) / 2];
    for (let i = 0; i < vertexCount; i++) for (let d = 0; d < 3; d++) world[i * 3 + d] = world[i * 3 + d]! - shift[d]!;
    const size3 = [0, 1, 2].map((d) => hi[d]! - lo[d]!);
    console.log('shoe size (m):', size3.map((v) => +v.toFixed(3)).join(' x '));
    for (const [measured, expected] of [[size3[0]!, SHOE.length], [size3[2]!, SHOE.width], [size3[1]!, SHOE.height]] as const) {
        if (Math.abs(measured - expected) > 0.001) throw new Error(`src/shoe/dimensions.ts says ${expected} m but the model is ${measured.toFixed(4)} m there`);
    }

    const { of: islandOf, count } = islandsOf(indices, vertexCount);
    if (count !== ISLANDS) throw new Error(`expected ${ISLANDS} islands, found ${count}`);
    const partOfIsland = new Array<string>(ISLANDS);
    for (const [part, from, to] of RANGES) {
        if (!(PARTS as readonly string[]).includes(part)) throw new Error(`${part} is not a part the app knows (src/parts.ts)`);
        for (let i = from; i <= to; i++) partOfIsland[i] = part;
    }
    if (partOfIsland.some((p) => !p)) throw new Error('an island has no part');

    // Colour textures: the grey detail map and each part's average colour per variant.
    const variants = src.getRoot().listMaterials().map((material) => ({ name: material.getBaseColorTexture()!.getName().replace(/^diffuse|\.jpg$/g, ''), texture: material.getBaseColorTexture()! }));
    const first = await rgbOf(variants[0]!.texture);
    const size = first.size;
    const { owner, covered } = rasterize(uv, indices, islandOf, size);
    const detail = new Float32Array(size * size);
    const colours: Record<string, Record<string, string>> = {};
    for (const variant of variants) {
        const { data, size: s } = await rgbOf(variant.texture);
        if (s !== size) throw new Error('textures differ in size');
        const sumY = new Float64Array(ISLANDS);
        const n = new Float64Array(ISLANDS);
        const lum = new Float32Array(size * size);
        const partRgb = new Map<string, number[]>();
        for (let i = 0; i < size * size; i++) {
            const r = srgbToLinear[data[i * 3]!]!;
            const g = srgbToLinear[data[i * 3 + 1]!]!;
            const b = srgbToLinear[data[i * 3 + 2]!]!;
            const y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            lum[i] = y;
            if (covered[i]) {
                sumY[owner[i]!] = sumY[owner[i]!]! + y;
                n[owner[i]!] = n[owner[i]!]! + 1;
                const acc = partRgb.get(partOfIsland[owner[i]!]!) ?? [0, 0, 0, 0];
                acc[0]! += r;
                acc[1]! += g;
                acc[2]! += b;
                acc[3]! += 1;
                partRgb.set(partOfIsland[owner[i]!]!, acc);
            }
        }
        colours[variant.name] = Object.fromEntries(PARTS.map((p) => [p, hex(partRgb.get(p)!.slice(0, 3).map((v) => v / partRgb.get(p)![3]!))]));
        for (let i = 0; i < size * size; i++) {
            const o = owner[i]!;
            const mean = o >= 0 && n[o]! > 0 ? sumY[o]! / n[o]! : 0;
            // Half of full brightness is "this part's colour"; the run-time multiplies it back by two.
            detail[i] = detail[i]! + (mean > 0 ? Math.min(1, (0.5 * lum[i]!) / mean) : 0.5) / variants.length;
        }
    }
    const grey = Buffer.alloc(size * size);
    for (let i = 0; i < size * size; i++) grey[i] = Math.round(Math.min(1, linearToSrgb(detail[i]!)) * 255);
    const detailPng = await sharp(grey, { raw: { width: size, height: size, channels: 1 } }).png().toBuffer();
    console.log('average colours per part, per variant:', JSON.stringify(colours, null, 2));

    // The output document: one mesh per part, sharing three textures.
    const doc = new Document();
    doc.getRoot().getAsset().generator = 'prepare-shoe.ts';
    doc.getRoot().getAsset().copyright = 'Shoe model: Shopify, CC BY 4.0 (Khronos glTF-Sample-Assets, MaterialsVariantsShoe)';
    const buffer = doc.createBuffer();
    const clearcoat = doc.createExtension(KHRMaterialsClearcoat);
    const srcMaterial = src.getRoot().listMaterials()[0]!;
    const normalTexture = doc.createTexture('normal').setImage(srcMaterial.getNormalTexture()!.getImage()!).setMimeType(srcMaterial.getNormalTexture()!.getMimeType());
    const ormTexture = doc.createTexture('orm').setImage(srcMaterial.getMetallicRoughnessTexture()!.getImage()!).setMimeType(srcMaterial.getMetallicRoughnessTexture()!.getMimeType());
    const detailTexture = doc.createTexture('detail').setImage(new Uint8Array(detailPng)).setMimeType('image/png');
    const scene = doc.createScene('shoe');
    for (const part of PARTS) {
        const keep: number[] = [];
        for (let t = 0; t < indices.length; t += 3) if (partOfIsland[islandOf[indices[t]!]!] === part) keep.push(indices[t]!, indices[t + 1]!, indices[t + 2]!);
        const remap = new Map<number, number>();
        for (const v of keep) if (!remap.has(v)) remap.set(v, remap.size);
        const p = new Float32Array(remap.size * 3);
        const nrm = new Float32Array(remap.size * 3);
        const t = new Float32Array(remap.size * 2);
        for (const [from, to] of remap) {
            p.set(world.subarray(from * 3, from * 3 + 3), to * 3);
            nrm.set(normal.subarray(from * 3, from * 3 + 3), to * 3);
            t.set(uv.subarray(from * 2, from * 2 + 2), to * 2);
        }
        const material = doc
            .createMaterial(part)
            .setBaseColorTexture(detailTexture)
            .setNormalTexture(normalTexture)
            .setOcclusionTexture(ormTexture)
            .setMetallicRoughnessTexture(ormTexture)
            .setRoughnessFactor(1)
            .setMetallicFactor(1)
            .setExtension('KHR_materials_clearcoat', clearcoat.createClearcoat().setClearcoatFactor(0).setClearcoatRoughnessFactor(0.06));
        const primitive = doc
            .createPrimitive()
            .setAttribute('POSITION', doc.createAccessor().setType('VEC3').setArray(p).setBuffer(buffer))
            .setAttribute('NORMAL', doc.createAccessor().setType('VEC3').setArray(nrm).setBuffer(buffer))
            .setAttribute('TEXCOORD_0', doc.createAccessor().setType('VEC2').setArray(t).setBuffer(buffer))
            .setIndices(doc.createAccessor().setType('SCALAR').setArray(Uint16Array.from(keep.map((v) => remap.get(v)!))).setBuffer(buffer))
            .setMaterial(material);
        const mesh = doc.createMesh(part).addPrimitive(primitive);
        scene.addChild(doc.createNode(part).setMesh(mesh));
        console.log(part.padEnd(9), String(remap.size).padStart(6), 'vertices', String(keep.length / 3).padStart(6), 'triangles');
    }

    await doc.transform(
        textureCompress({ encoder: sharp, targetFormat: 'webp', resize: [2048, 2048], slots: /baseColorTexture/, quality: 88 }),
        textureCompress({ encoder: sharp, targetFormat: 'webp', resize: [1024, 1024], slots: /normalTexture/, quality: 92 }),
        textureCompress({ encoder: sharp, targetFormat: 'webp', resize: [1024, 1024], slots: /(occlusion|metallicRoughness)Texture/, quality: 85 }),
        meshopt({ encoder: MeshoptEncoder, level: 'high' }),
    );
    mkdirSync(dirname(output), { recursive: true });
    writeFileSync(shadowOutput, await contactShadow(world, indices));
    const glb = await io.writeBinary(doc);
    writeFileSync(output, glb);
    console.log(`wrote ${output}: ${(glb.byteLength / 1024).toFixed(0)} kB`);
}

await main();
