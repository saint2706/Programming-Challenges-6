import { TraceMap, originalPositionFor } from '@jridgewell/trace-mapping';

export interface Frame {
    file: string;
    /** 1-based, as in a stack trace. */
    line: number;
    column: number;
}

export interface Position {
    file: string;
    line: number;
    column: number;
}

/** The name the sandbox gives the run's bundle in stack traces (it replaces the blob: URL). */
export const BUNDLE = 'bundle.js';

/** Pull `file:line:column` frames out of a V8 or Firefox stack. Lines that are not frames are skipped. */
export function parseFrames(stack: string): Frame[] {
    const frames: Frame[] = [];
    for (const raw of stack.split('\n')) {
        const m = /(?:\(|@|\bat |^\s*)([^\s()@]+?):(\d+):(\d+)\)?\s*$/.exec(raw.trim());
        if (m) frames.push({ file: m[1]!, line: Number(m[2]), column: Number(m[3]) });
    }
    return frames;
}

/**
 * The name esbuild's map gives a source, as the person knows it: `pg:/main.tsx` is `main.tsx`, and a
 * built-in package stays `vendor:preact/hooks` so it is never mistaken for one of their files.
 */
export function fileName(source: string): string {
    const vendor = /(?:^|\/)(vendor:.+)$/.exec(source);
    if (vendor) return vendor[1]!;
    return source.replace(/^.*?pg:/, '').replace(/^.*\//, '');
}

/** Maps positions in the running bundle back to the file the person typed. */
export class Mapper {
    private readonly trace: TraceMap | null;

    constructor(map: string | null) {
        try {
            this.trace = map ? new TraceMap(map) : null;
        } catch {
            this.trace = null;
        }
    }

    /** `line` is 1-based and `column` 1-based, as in a stack frame. */
    original(line: number, column: number): Position | null {
        // Stacks come from the preview, which can write whatever it likes into them: `bundle.js:0:0` must not throw.
        if (!this.trace || !Number.isInteger(line) || line < 1 || !Number.isInteger(column)) return null;
        try {
            const p = originalPositionFor(this.trace, { line, column: Math.max(0, column - 1) });
            if (p.source === null || p.line === null) return null;
            return { file: fileName(p.source), line: p.line, column: (p.column ?? 0) + 1 };
        } catch {
            return null;
        }
    }

    /** The first frame of `stack` that is in the bundle, mapped back to a file. */
    firstOriginal(stack: string): Position | null {
        for (const f of parseFrames(stack)) {
            if (f.file !== BUNDLE) continue;
            const p = this.original(f.line, f.column);
            if (p) return p;
        }
        return null;
    }

    /** Rewrite every bundle frame in a stack to the original file, so people read their own code. */
    rewriteStack(stack: string): string {
        return stack
            .split('\n')
            .map((raw) => {
                const frames = parseFrames(raw);
                const f = frames[0];
                if (!f || f.file !== BUNDLE) return raw;
                const p = this.original(f.line, f.column);
                return p ? raw.replace(`${BUNDLE}:${f.line}:${f.column}`, `${p.file}:${p.line}:${p.column}`) : raw;
            })
            .join('\n');
    }
}
