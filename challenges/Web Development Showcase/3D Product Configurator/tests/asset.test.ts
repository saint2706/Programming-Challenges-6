import { createHash } from 'node:crypto';
import { readFileSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { type Document, NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { getBounds } from '@gltf-transform/functions';
import { MeshoptDecoder } from 'meshoptimizer';
import sharp from 'sharp';
import { beforeAll, describe, expect, test } from 'vitest';
import { PART_IDS } from '../src/parts.ts';
import { SHADOW, SHOE } from '../src/shoe/dimensions.ts';

const file = (name: string) => fileURLToPath(new URL(`../public/${name}`, import.meta.url));

describe('the shoe model the build leaves behind (public/models/shoe.glb)', () => {
    let doc: Document;
    beforeAll(async () => {
        await MeshoptDecoder.ready;
        doc = await new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({ 'meshopt.decoder': MeshoptDecoder }).read(file('models/shoe.glb'));
    });

    test('is one mesh per part, each with its own material and one primitive, and nothing else', () => {
        const root = doc.getRoot();
        expect(root.listMeshes().map((m) => m.getName()).sort()).toEqual([...PART_IDS].sort());
        expect(root.listMaterials().map((m) => m.getName()).sort()).toEqual([...PART_IDS].sort());
        for (const mesh of root.listMeshes()) {
            expect(mesh.listPrimitives(), mesh.getName()).toHaveLength(1);
            const prim = mesh.listPrimitives()[0]!;
            expect(prim.getMaterial()?.getName()).toBe(mesh.getName());
            for (const attribute of ['POSITION', 'NORMAL', 'TEXCOORD_0']) expect(prim.getAttribute(attribute), `${mesh.getName()} ${attribute}`).not.toBeNull();
            expect(prim.getIndices()!.getCount() % 3).toBe(0);
        }
        expect(root.listNodes().map((n) => n.getName()).sort()).toEqual([...PART_IDS].sort());
    });

    test('every part has real geometry, and the parts together are the whole shoe', () => {
        let triangles = 0;
        for (const mesh of doc.getRoot().listMeshes()) {
            const t = mesh.listPrimitives()[0]!.getIndices()!.getCount() / 3;
            expect(t, mesh.getName()).toBeGreaterThan(300);
            triangles += t;
        }
        expect(triangles).toBe(22700); // what the Khronos model has: nothing lost, nothing added, in the split
    });

    test('is the size, and stands where, src/shoe/dimensions.ts and the scene expect: metres, on the floor, centred', () => {
        const { min, max } = getBounds(doc.getRoot().listScenes()[0]!);
        expect(max[0]! - min[0]!).toBeCloseTo(SHOE.length, 2);
        expect(max[1]! - min[1]!).toBeCloseTo(SHOE.height, 2);
        expect(max[2]! - min[2]!).toBeCloseTo(SHOE.width, 2);
        expect(min[1]!).toBeCloseTo(0, 3);
        expect((max[0]! + min[0]!) / 2).toBeCloseTo(0, 3);
        expect((max[2]! + min[2]!) / 2).toBeCloseTo(0, 3);
    });

    test('shares three textures between all parts, as WebP, at sizes the page can afford', () => {
        const textures = doc.getRoot().listTextures();
        expect(textures.map((t) => t.getName()).sort()).toEqual(['detail', 'normal', 'orm']);
        for (const t of textures) expect(t.getMimeType()).toBe('image/webp');
        for (const m of doc.getRoot().listMaterials()) {
            expect(m.getBaseColorTexture()?.getName()).toBe('detail');
            expect(m.getNormalTexture()?.getName()).toBe('normal');
            expect(m.getMetallicRoughnessTexture()?.getName()).toBe('orm');
            expect(m.getOcclusionTexture()?.getName()).toBe('orm');
        }
    });

    test('has the clearcoat extension on every material (at zero), so the page can switch a finish on without rebuilding materials', () => {
        expect(doc.getRoot().listExtensionsUsed().map((e) => e.extensionName)).toContain('KHR_materials_clearcoat');
        for (const m of doc.getRoot().listMaterials()) {
            const cc = m.getExtension('KHR_materials_clearcoat') as unknown as { getClearcoatFactor(): number } | null;
            expect(cc, m.getName()).not.toBeNull();
            expect(cc!.getClearcoatFactor()).toBe(0);
        }
    });

    test('has texture coordinates inside the textures, and indices inside the vertices', () => {
        for (const mesh of doc.getRoot().listMeshes()) {
            const prim = mesh.listPrimitives()[0]!;
            const uv = prim.getAttribute('TEXCOORD_0')!;
            const e = [0, 0];
            for (let i = 0; i < uv.getCount(); i++) {
                uv.getElement(i, e);
                expect(e[0]!).toBeGreaterThanOrEqual(-1e-3);
                expect(e[0]!).toBeLessThanOrEqual(1 + 1e-3);
                expect(e[1]!).toBeGreaterThanOrEqual(-1e-3);
                expect(e[1]!).toBeLessThanOrEqual(1 + 1e-3);
            }
            const indices = prim.getIndices()!;
            const count = prim.getAttribute('POSITION')!.getCount();
            for (let i = 0; i < indices.getCount(); i++) expect(indices.getScalar(i)).toBeLessThan(count);
        }
    });

    test('is small enough to ship', () => {
        expect(statSync(file('models/shoe.glb')).size).toBeLessThan(1.5 * 1024 * 1024);
    });
});

describe('the other assets', () => {
    test('the shadow picture is the size the page puts on the floor, dark at the edges and darkest under the shoe', async () => {
        const { data, info } = await sharp(file('models/shadow.png')).greyscale().raw().toBuffer({ resolveWithObject: true });
        expect(info.width).toBe(SHADOW.pixels);
        expect(info.height).toBe(SHADOW.pixels);
        const at = (x: number, y: number) => data[y * info.width + x]!;
        expect(at(2, 2)).toBe(0);
        expect(at(info.width - 3, info.height - 3)).toBe(0);
        expect(at(info.width / 2, info.height / 2)).toBeGreaterThan(150);
    });

    test('the face model and the lighting are the files they were taken from (checked by hash)', () => {
        const model = readFileSync(file('models/face_landmarker.task'));
        expect(createHash('sha256').update(model).digest('hex')).toBe('64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff');
        const hdr = readFileSync(file('env/studio_small_08_1k.hdr'));
        expect(hdr.subarray(0, 10).toString('latin1')).toBe('#?RADIANCE');
        expect(createHash('sha256').update(hdr).digest('hex')).toBe('f6a989f89432eb4eee3191364a9c1ceed195c4ec3544173a3c04fd96cb91d0ba');
    });
});
