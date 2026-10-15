import type * as Esbuild from 'esbuild-wasm';
import { GUARD_RUNTIME } from './guardruntime.ts';
import { guardLoops } from './loopguard.ts';
import { type Project, entryOf, kindOf } from './project.ts';
import { fileName } from './sourcemap.ts';

export interface Diagnostic {
    file: string;
    /** 1-based line and column; `length` characters long (at least 1). */
    line: number;
    column: number;
    length: number;
    message: string;
    severity: 'error' | 'warning';
}

export type BundleResult =
    | { ok: true; code: string; map: string | null; warnings: Diagnostic[]; loops: number }
    | { ok: false; errors: Diagnostic[] };

export type EsbuildLike = Pick<typeof Esbuild, 'build'>;

/** Packages the playground carries itself: source text keyed by the specifier a person writes. */
export type Vendor = Readonly<Record<string, string>>;

export const ESM_ORIGIN = 'https://esm.sh';

/** A bare package specifier, with an optional version and subpath. Nothing that could be a URL trick. */
const PACKAGE = /^(@[a-z0-9~-][a-z0-9._~-]*\/)?[a-z0-9~-][a-z0-9._~-]*(@[A-Za-z0-9._~^<>=*+|-]+)?(\/[A-Za-z0-9._~@+-]+)*$/;

const TRY_EXTENSIONS = ['.ts', '.tsx', '.js', '.jsx', '.mjs', '.json'];

const LOADERS: Record<string, Esbuild.Loader> = { ts: 'ts', tsx: 'tsx', js: 'js', mjs: 'js', jsx: 'jsx', json: 'json' };

/** Find the file a relative import means. Folders do not exist, so only `./name` is allowed. */
export function resolveRelative(spec: string, names: readonly string[]): string | null {
    const m = /^\.\/([^/]+)$/.exec(spec);
    if (!m) return null;
    const want = m[1]!;
    const has = (n: string) => names.find((x) => x === n) ?? null;
    const direct = has(want);
    if (direct) return direct;
    for (const ext of TRY_EXTENSIONS) {
        const found = has(want + ext);
        if (found) return found;
    }
    // TypeScript habit: write `./a.js` for a file that is really `a.ts`.
    const swap = /^(.*)\.(js|jsx|mjs)$/.exec(want);
    if (swap) for (const ext of swap[2] === 'jsx' ? ['.tsx'] : ['.ts', '.tsx']) {
        const found = has(swap[1]! + ext);
        if (found) return found;
    }
    return null;
}

export async function bundleProject(project: Project, esbuild: EsbuildLike, vendor: Vendor): Promise<BundleResult> {
    const entry = entryOf(project);
    if (!entry) return { ok: true, code: '', map: null, warnings: [], loops: 0 };

    const files = new Map(project.files.map((f) => [f.name, f.code]));
    const names = [...files.keys()];
    const vendored = Object.keys(vendor).join(', ');
    const notes: Diagnostic[] = [];

    const plugin: Esbuild.Plugin = {
        name: 'playground',
        setup(build) {
            build.onResolve({ filter: /.*/ }, (args) => {
                if (args.kind === 'entry-point') return { path: `/${entry.name}`, namespace: 'pg' };
                const spec = args.path;
                if (spec.startsWith('./') || spec.startsWith('../')) {
                    if (args.namespace !== 'pg') return { errors: [{ text: `Cannot import ${spec} from a built-in package.` }] };
                    const found = resolveRelative(spec, names);
                    if (found) return { path: `/${found}`, namespace: 'pg' };
                    return { errors: [{ text: `Could not find ${spec}. Files live in one flat folder, so write ./name and make sure the file exists.` }] };
                }
                if (Object.hasOwn(vendor, spec)) return { path: spec, namespace: 'vendor' };
                if (/^https?:\/\//i.test(spec)) {
                    if (!project.network) return { errors: [{ text: `Importing from a URL needs "Allow network". It is off.` }] };
                    if (!spec.startsWith(`${ESM_ORIGIN}/`) || /[\s"'<>\\]/.test(spec)) return { errors: [{ text: `Only ${ESM_ORIGIN} may be imported by URL.` }] };
                    return { path: spec, external: true };
                }
                if (PACKAGE.test(spec) && !spec.includes('..')) {
                    if (project.network) return { path: `${ESM_ORIGIN}/${spec}`, external: true };
                    return {
                        errors: [{ text: `Cannot find package "${spec}". Built in: ${vendored}. Turn on "Allow network" to load other packages from ${ESM_ORIGIN}.` }],
                    };
                }
                return { errors: [{ text: `"${spec}" is not an import this playground can resolve.` }] };
            });

            build.onLoad({ filter: /.*/, namespace: 'pg' }, (args) => {
                const name = args.path.slice(1);
                const code = files.get(name);
                if (code === undefined) return { errors: [{ text: `No file called ${name}.` }] };
                const kind = kindOf(name);
                if (kind === 'css' || kind === 'html') {
                    notes.push({ file: name, line: 1, column: 1, length: 1, severity: 'warning', message: `${name} is added to the page on its own; importing it from a script does nothing.` });
                    return { contents: '', loader: 'js' };
                }
                const extension = name.slice(name.lastIndexOf('.') + 1).toLowerCase();
                return { contents: code, loader: Object.hasOwn(LOADERS, extension) ? LOADERS[extension]! : 'js' };
            });

            build.onLoad({ filter: /.*/, namespace: 'vendor' }, (args) => ({ contents: vendor[args.path]!, loader: 'js' }));
        },
    };

    let result: Esbuild.BuildResult;
    try {
        result = await esbuild.build({
            entryPoints: ['playground-entry'],
            bundle: true,
            write: false,
            outfile: 'bundle.js',
            format: 'esm',
            platform: 'browser',
            target: 'es2022',
            sourcemap: 'external',
            jsx: 'automatic',
            jsxImportSource: 'preact',
            logLevel: 'silent',
            legalComments: 'none',
            // The loop guard's own definition goes first; as a banner, esbuild keeps its source map exact.
            banner: { js: GUARD_RUNTIME },
            plugins: [plugin],
        });
    } catch (e) {
        const failure = e as Partial<Esbuild.BuildFailure>;
        if (failure.errors) return { ok: false, errors: failure.errors.map((m) => diagnostic(m, 'error')) };
        throw e;
    }

    const js = result.outputFiles?.find((f) => f.path.endsWith('.js'));
    const map = result.outputFiles?.find((f) => f.path.endsWith('.map'));
    if (!js) return { ok: false, errors: [{ file: entry.name, line: 1, column: 1, length: 1, message: 'The build produced nothing.', severity: 'error' }] };
    const warnings = [...notes, ...result.warnings.map((m) => diagnostic(m, 'warning'))];

    // With `sourcemap: 'external'` esbuild writes the map as a separate file and leaves no pointer to it
    // in the code (a test checks this), so the code can be used as it is.
    const code = js.text;
    try {
        const guarded = guardLoops(code, map?.text ?? null);
        return { ok: true, code: guarded.code, map: guarded.map, warnings, loops: guarded.loops };
    } catch {
        warnings.push({ file: entry.name, line: 1, column: 1, length: 1, severity: 'warning', message: 'Loop protection could not be applied to this build, so an infinite loop would freeze the page.' });
        return { ok: true, code, map: map?.text ?? null, warnings, loops: 0 };
    }
}

function diagnostic(m: Esbuild.Message, severity: 'error' | 'warning'): Diagnostic {
    const loc = m.location;
    return {
        file: loc ? fileName(loc.file) : '',
        line: loc?.line ?? 1,
        column: (loc?.column ?? 0) + 1,
        length: Math.max(1, loc?.length ?? 1),
        message: m.text,
        severity,
    };
}
