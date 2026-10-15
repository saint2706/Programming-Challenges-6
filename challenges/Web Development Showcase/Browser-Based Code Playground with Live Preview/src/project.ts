/** The thing a person edits: a few named files, all in one flat folder. */

export const MAX_FILES = 12;
export const MAX_FILE_BYTES = 200_000;
export const MAX_PROJECT_BYTES = 500_000;
export const MAX_NAME = 40;

export type Kind = 'html' | 'css' | 'script' | 'json';

export interface ProjectFile {
    name: string;
    code: string;
}

export interface Project {
    files: ProjectFile[];
    /** Whether the preview may use the network (https images and fetch, and packages from esm.sh). */
    network: boolean;
}

const EXTENSIONS: Record<string, Kind> = {
    html: 'html',
    css: 'css',
    js: 'script',
    mjs: 'script',
    jsx: 'script',
    ts: 'script',
    tsx: 'script',
    json: 'json',
};

export const HTML_FILE = 'index.html';

export function extensionOf(name: string): string {
    const dot = name.lastIndexOf('.');
    return dot < 0 ? '' : name.slice(dot + 1).toLowerCase();
}

export function kindOf(name: string): Kind | null {
    const extension = extensionOf(name);
    return Object.hasOwn(EXTENSIONS, extension) ? EXTENSIONS[extension]! : null;
}

/** Why `name` cannot be a file name here, or `null` when it can. Flat names only: no folders, no dot-files. */
export function nameProblem(name: string): string | null {
    if (name.length === 0) return 'Give the file a name.';
    if (name.length > MAX_NAME) return `Names can be at most ${MAX_NAME} characters.`;
    if (!/^[A-Za-z0-9_-][A-Za-z0-9._-]*$/.test(name)) return 'Use letters, digits, dot, dash and underscore only (no folders).';
    if (!kindOf(name)) return 'Files can end in .html, .css, .js, .mjs, .jsx, .ts, .tsx or .json.';
    return null;
}

/** Is this the file the page's script starts from: `main.js`, `main.ts`, `main.tsx` and so on? */
export function isEntry(name: string): boolean {
    return /^main\.(js|mjs|jsx|ts|tsx)$/i.test(name);
}

export function entryOf(project: Project): ProjectFile | undefined {
    return project.files.find((f) => isEntry(f.name));
}

export function totalBytes(files: readonly ProjectFile[]): number {
    return files.reduce((n, f) => n + new Blob([f.code]).size, 0);
}

/** Add, rename or delete must keep the project valid; this says why a candidate set of files is not. */
export function filesProblem(files: readonly ProjectFile[]): string | null {
    if (files.length > MAX_FILES) return `At most ${MAX_FILES} files.`;
    const seen = new Set<string>();
    let html = 0;
    let entries = 0;
    for (const f of files) {
        const problem = nameProblem(f.name);
        if (problem) return `${f.name || '(empty)'}: ${problem}`;
        const key = f.name.toLowerCase();
        if (seen.has(key)) return `Two files are called ${f.name}.`;
        seen.add(key);
        if (kindOf(f.name) === 'html') {
            html++;
            if (f.name !== HTML_FILE) return `The page must be called ${HTML_FILE}.`;
        }
        if (isEntry(f.name)) entries++;
        if (typeof f.code !== 'string') return `${f.name} has no text.`;
        if (new Blob([f.code]).size > MAX_FILE_BYTES) return `${f.name} is over ${MAX_FILE_BYTES / 1000} kB.`;
    }
    if (html !== 1) return `There must be exactly one ${HTML_FILE}.`;
    if (entries > 1) return 'Only one main script (main.js, main.ts, ...) is allowed.';
    if (totalBytes(files) > MAX_PROJECT_BYTES) return `The project is over ${MAX_PROJECT_BYTES / 1000} kB.`;
    return null;
}

/**
 * Turn untrusted data (a shared link, saved storage) into a project, or `null` when it is not a valid
 * one. Nothing from outside is trusted: every name, size and count is checked again here.
 */
export function parseProject(value: unknown): Project | null {
    if (typeof value !== 'object' || value === null) return null;
    const v = value as { files?: unknown; network?: unknown };
    if (!Array.isArray(v.files) || v.files.length === 0 || v.files.length > MAX_FILES) return null;
    const files: ProjectFile[] = [];
    for (const f of v.files) {
        if (typeof f !== 'object' || f === null) return null;
        const { name, code } = f as { name?: unknown; code?: unknown };
        if (typeof name !== 'string' || typeof code !== 'string') return null;
        files.push({ name, code });
    }
    if (filesProblem(files)) return null;
    return { files, network: v.network === true };
}

export function clone(project: Project): Project {
    return { network: project.network, files: project.files.map((f) => ({ ...f })) };
}

/** A name for a new file that is not taken: `untitled.js`, `untitled-2.js`, ... */
export function freshName(files: readonly ProjectFile[], base = 'untitled', ext = 'js'): string {
    const taken = new Set(files.map((f) => f.name.toLowerCase()));
    for (let i = 1; ; i++) {
        const name = `${base}${i === 1 ? '' : `-${i}`}.${ext}`;
        if (!taken.has(name.toLowerCase())) return name;
    }
}
