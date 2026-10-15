import { parse } from 'acorn';
import remapping from '@jridgewell/remapping';
import { TraceMap, originalPositionFor } from '@jridgewell/trace-mapping';
import MagicString from 'magic-string';

/** The call put at the start of every loop body. The sandbox defines it (`public/sandbox.js`). */
export const GUARD = '__pgLoop();';

interface Node {
    type: string;
    start: number;
    end: number;
    loc: { start: { line: number; column: number } };
    [key: string]: unknown;
}

const LOOPS = new Set(['ForStatement', 'WhileStatement', 'DoWhileStatement', 'ForInStatement', 'ForOfStatement']);

export interface Guarded {
    code: string;
    /** Source map from the guarded code back to the original files (esbuild's map, composed with ours). */
    map: string | null;
    loops: number;
}

/**
 * Put a guard call at the start of every loop body.
 *
 * An infinite loop in the preview freezes the whole tab (measured: Chrome runs a sandboxed iframe on
 * the page's own thread), so it has to be stopped from inside. The guard costs an increment and a
 * mask test per iteration and checks the clock every few iterations; the sandbox throws once one
 * synchronous run has lasted too long. `code` must be plain modern JavaScript, which is what esbuild
 * emits after TypeScript and JSX have been removed.
 *
 * Only loops that come from the person's own files are guarded: the bundled packages are trusted,
 * and a guard in the middle of Preact's diffing would only slow it down. Where a loop came from is
 * read from `map` (esbuild's source map for `code`, as a JSON string); with no map, every loop is
 * guarded. The result's map goes straight back to the original sources, with columns kept exact by
 * magic-string's edit-aware map.
 */
export function guardLoops(code: string, map: string | null): Guarded {
    const ast = parse(code, { ecmaVersion: 'latest', sourceType: 'module', allowAwaitOutsideFunction: true, locations: true }) as unknown as Node;
    const edit = new MagicString(code);
    const trace = map ? new TraceMap(map) : null;
    const isUsers = (node: Node): boolean => {
        if (!trace) return true;
        const from = originalPositionFor(trace, { line: node.loc.start.line, column: node.loc.start.column });
        return from.source === null || from.source.startsWith('pg:'); // unknown origin: guard it, to be safe
    };
    let loops = 0;

    const visit = (node: Node): void => {
        if (LOOPS.has(node.type) && isUsers(node)) {
            const body = node.body as Node;
            loops++;
            if (body.type === 'BlockStatement') {
                edit.appendLeft(body.start + 1, GUARD);
            } else {
                // `while (x) y;` has no braces to put the call in, so add them.
                edit.appendLeft(body.start, `{${GUARD}`);
                edit.appendLeft(body.end, '}');
            }
        }
        for (const key in node) {
            if (key === 'type' || key === 'start' || key === 'end') continue;
            const child = node[key];
            if (Array.isArray(child)) {
                for (const c of child) if (isNode(c)) visit(c);
            } else if (isNode(child)) visit(child);
        }
    };
    visit(ast);

    if (loops === 0) return { code, map, loops };
    const guarded = edit.toString();
    if (map === null) return { code: guarded, map: null, loops };
    const mine = edit.generateMap({ hires: 'boundary', includeContent: false });
    const composed = remapping([mine as never, map], () => null);
    return { code: guarded, map: composed.toString(), loops };
}

function isNode(value: unknown): value is Node {
    return typeof value === 'object' && value !== null && typeof (value as Node).type === 'string';
}
