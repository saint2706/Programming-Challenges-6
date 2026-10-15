import fc from 'fast-check';
import { describe, expect, test } from 'vitest';
import type { Project } from '../src/project.ts';
import { MAX_FRAGMENT_CHARS, MAX_INFLATED_BYTES, decodeProject, encodeProject } from '../src/share.ts';
import { TEMPLATES } from '../src/templates.ts';

const project = (code: string, network = false): Project => ({ network, files: [{ name: 'index.html', code }, { name: 'main.js', code: 'console.log(1)' }] });

async function deflateRaw(bytes: Uint8Array): Promise<Uint8Array> {
    return new Uint8Array(await new Response(new Blob([bytes as BlobPart]).stream().pipeThrough(new CompressionStream('deflate-raw'))).arrayBuffer());
}
const b64url = (bytes: Uint8Array) => Buffer.from(bytes).toString('base64url');

describe('round trip', () => {
    test('every template survives, with the # or without', async () => {
        for (const t of TEMPLATES) {
            const fragment = await encodeProject(t.project);
            expect(await decodeProject(`#${fragment}`)).toEqual(t.project);
            expect(await decodeProject(fragment)).toEqual(t.project);
        }
    });

    test('any text, including emoji, newlines, quotes and lone surrogates that JSON escapes', async () => {
        await fc.assert(
            fc.asyncProperty(fc.string({ unit: 'binary', maxLength: 600 }), fc.boolean(), async (code, network) => {
                const p = project(code, network);
                expect(await decodeProject(await encodeProject(p))).toEqual(JSON.parse(JSON.stringify(p)));
            }),
            { numRuns: 60 },
        );
    });

    test('a big project becomes a link of reasonable length', async () => {
        const fragment = await encodeProject(project('<p>line of text that repeats</p>\n'.repeat(2000)));
        expect(fragment.length).toBeLessThan(2000); // repeated text compresses well
        expect(fragment.startsWith('p=v1.')).toBe(true);
        expect(fragment).toMatch(/^[A-Za-z0-9_.=-]+$/); // safe in a URL fragment
    });
});

describe('hostile fragments never throw and never produce a project', () => {
    test.each([
        ['empty', ''],
        ['no prefix', 'hello'],
        ['wrong version', 'p=v2.AAAA'],
        ['not base64', 'p=v1.!!!!'],
        ['base64 that is not deflate', `p=v1.${b64url(Buffer.from('this is not compressed data at all'))}`],
        ['too long', `p=v1.${'A'.repeat(MAX_FRAGMENT_CHARS + 1)}`],
    ])('%s', async (_label, fragment) => {
        expect(await decodeProject(fragment)).toBeNull();
    });

    test('valid compressed JSON that is not a project', async () => {
        for (const json of ['null', '[]', '{"files":[]}', '{"files":[{"name":"../x","code":""}]}', '"text"', '{"files":[{"name":"index.html","code":1}]}']) {
            const compressed = await deflateRaw(new TextEncoder().encode(json));
            expect(await decodeProject(`p=v1.${b64url(compressed)}`), json).toBeNull();
        }
    });

    test('invalid UTF-8 inside valid compressed data', async () => {
        const compressed = await deflateRaw(Uint8Array.from([0x7b, 0xff, 0xfe, 0x7d]));
        expect(await decodeProject(`p=v1.${b64url(compressed)}`)).toBeNull();
    });

    test('a decompression bomb is cut off instead of expanded', async () => {
        // 20 MB of zeros compresses to a few kilobytes; it must not be inflated in full.
        const bomb = await deflateRaw(new Uint8Array(20_000_000));
        expect(bomb.length).toBeLessThan(MAX_FRAGMENT_CHARS);
        const before = performance.now();
        expect(await decodeProject(`p=v1.${b64url(bomb)}`)).toBeNull();
        expect(performance.now() - before).toBeLessThan(2000);
        expect(MAX_INFLATED_BYTES).toBeLessThan(20_000_000);
    });

    test('never throws on arbitrary strings', async () => {
        await fc.assert(fc.asyncProperty(fc.string({ maxLength: 200 }), async (s) => void (await decodeProject(s))), { numRuns: 100 });
        await fc.assert(fc.asyncProperty(fc.string({ maxLength: 200 }), async (s) => void (await decodeProject(`p=v1.${s}`))), { numRuns: 100 });
    });
});
