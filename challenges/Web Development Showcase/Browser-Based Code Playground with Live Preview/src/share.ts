import { type Project, parseProject } from './project.ts';

/** A project in the address bar: `#p=v1.<deflated JSON as base64url>`. Nothing is ever sent to a server. */
const PREFIX = 'p=v1.';
/** Longest fragment we will even try to read. */
export const MAX_FRAGMENT_CHARS = 250_000;
/** Longest the JSON may be after inflating: a link must not be able to expand into gigabytes (a zip bomb). */
export const MAX_INFLATED_BYTES = 1_000_000;

function toBase64Url(bytes: Uint8Array): string {
    let binary = '';
    for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function fromBase64Url(text: string): Uint8Array | null {
    if (!/^[A-Za-z0-9_-]*$/.test(text)) return null;
    try {
        const binary = atob(text.replace(/-/g, '+').replace(/_/g, '/'));
        return Uint8Array.from(binary, (c) => c.charCodeAt(0));
    } catch {
        return null;
    }
}

async function deflate(bytes: Uint8Array): Promise<Uint8Array> {
    const stream = new Blob([bytes as BlobPart]).stream().pipeThrough(new CompressionStream('deflate-raw'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Inflate, giving up as soon as the output passes `limit` bytes. */
async function inflate(bytes: Uint8Array, limit: number): Promise<Uint8Array | null> {
    const reader = new Blob([bytes as BlobPart]).stream().pipeThrough(new DecompressionStream('deflate-raw')).getReader();
    const chunks: Uint8Array[] = [];
    let total = 0;
    try {
        for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            total += value.length;
            if (total > limit) {
                await reader.cancel();
                return null;
            }
            chunks.push(value);
        }
    } catch {
        return null; // not valid deflate data
    }
    const out = new Uint8Array(total);
    let at = 0;
    for (const c of chunks) {
        out.set(c, at);
        at += c.length;
    }
    return out;
}

/** The fragment (without the `#`) that reproduces `project`. */
export async function encodeProject(project: Project): Promise<string> {
    const json = JSON.stringify({ network: project.network, files: project.files });
    return PREFIX + toBase64Url(await deflate(new TextEncoder().encode(json)));
}

/** The project a fragment describes, or `null` for anything that is not exactly that. Never throws. */
export async function decodeProject(fragment: string): Promise<Project | null> {
    const text = fragment.startsWith('#') ? fragment.slice(1) : fragment;
    if (!text.startsWith(PREFIX) || text.length > MAX_FRAGMENT_CHARS) return null;
    const compressed = fromBase64Url(text.slice(PREFIX.length));
    if (!compressed) return null;
    const inflated = await inflate(compressed, MAX_INFLATED_BYTES);
    if (!inflated) return null;
    try {
        return parseProject(JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(inflated)));
    } catch {
        return null;
    }
}
