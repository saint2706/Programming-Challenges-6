import type { BundleResult, Diagnostic } from './bundle.ts';
import type { RunPayload } from './preview.ts';
import { HTML_FILE, type Project, kindOf } from './project.ts';
import { Mapper } from './sourcemap.ts';

export type RunOutcome =
    /** Nothing that shows on the page changed. */
    | { kind: 'none' }
    /** Only stylesheets changed: swapped into the running page, which kept its state. */
    | { kind: 'css' }
    /** A new page was started. */
    | { kind: 'full'; run: number; warnings: Diagnostic[]; loops: number; ms: number }
    /** The build failed; the page that was running is untouched. */
    | { kind: 'failed'; errors: Diagnostic[] }
    /** A newer edit arrived while this one was building, so this result was dropped. */
    | { kind: 'stale' };

export interface PreviewPort {
    run(payload: RunPayload): number;
    updateCss(css: string): boolean;
    readonly live: boolean;
}

interface Snapshot {
    html: string;
    css: string;
    scripts: string;
    network: boolean;
}

/**
 * Decides what an edit needs. A change to stylesheets alone is swapped into the running page (it
 * keeps its DOM and JavaScript state); anything else rebuilds and starts a fresh page. Builds that
 * fail leave the running page alone, so a typo never blanks the preview.
 */
export class Runner {
    private readonly port: PreviewPort;
    private readonly build: (project: Project) => Promise<BundleResult>;
    private readonly now: () => number;
    private last: Snapshot | null = null;
    private seq = 0;
    private readonly mappers = new Map<number, Mapper>();

    constructor(port: PreviewPort, build: (project: Project) => Promise<BundleResult>, now: () => number = () => performance.now()) {
        this.port = port;
        this.build = build;
        this.now = now;
    }

    /** The source map of a run, to turn a position in the bundle back into the person's file. */
    mapperFor(run: number): Mapper | undefined {
        return this.mappers.get(run);
    }

    /** Forget what is on screen, so the next update starts a fresh page even if nothing changed. */
    invalidate(): void {
        this.last = null;
    }

    async update(project: Project, options: { force?: boolean } = {}): Promise<RunOutcome> {
        const snapshot = snapshotOf(project);
        const before = this.last;
        if (!options.force && before) {
            const sameShell = before.html === snapshot.html && before.scripts === snapshot.scripts && before.network === snapshot.network;
            if (sameShell && before.css === snapshot.css) return { kind: 'none' };
            if (sameShell && this.port.live && this.port.updateCss(snapshot.css)) {
                this.last = snapshot;
                return { kind: 'css' };
            }
        }

        const seq = ++this.seq;
        const started = this.now();
        const result = await this.build(project);
        if (seq !== this.seq) return { kind: 'stale' };
        if (!result.ok) return { kind: 'failed', errors: result.errors };

        const run = this.port.run({ html: snapshot.html, css: snapshot.css, code: result.code, network: snapshot.network });
        this.mappers.set(run, new Mapper(result.map));
        for (const old of [...this.mappers.keys()]) if (old < run - 3) this.mappers.delete(old);
        this.last = snapshot;
        return { kind: 'full', run, warnings: result.warnings, loops: result.loops, ms: Math.round(this.now() - started) };
    }
}

function snapshotOf(project: Project): Snapshot {
    const html = project.files.find((f) => f.name === HTML_FILE)?.code ?? '';
    const css = project.files.filter((f) => kindOf(f.name) === 'css').map((f) => f.code).join('\n');
    // Everything else is script or data. Sorted, so reordering tabs alone is not a change.
    const scripts = JSON.stringify(
        project.files
            .filter((f) => kindOf(f.name) !== 'html' && kindOf(f.name) !== 'css')
            .map((f) => [f.name, f.code])
            .sort((a, b) => (a[0]! < b[0]! ? -1 : 1)),
    );
    return { html, css, scripts, network: project.network };
}
