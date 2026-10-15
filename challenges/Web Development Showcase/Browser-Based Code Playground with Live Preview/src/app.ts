import type { BundleResult, Diagnostic } from './bundle.ts';
import { ConsolePanel, type Filter } from './console.ts';
import { $, h } from './dom.ts';
import { Editor } from './editor.ts';
import { loadEsbuild } from './esbuild.ts';
import type { SandboxEvent } from './messages.ts';
import { Preview } from './preview.ts';
import {
    HTML_FILE,
    type Project,
    clone,
    entryOf,
    extensionOf,
    filesProblem,
    freshName,
    isEntry,
    nameProblem,
} from './project.ts';
import { Runner } from './runner.ts';
import { decodeProject, encodeProject } from './share.ts';
import { clearBooting, didNotFinishBooting, loadSaved, markBooting, save } from './storage.ts';
import { DEFAULT_TEMPLATE, TEMPLATES } from './templates.ts';

const RUN_DELAY_MS = 300;
const SAVE_DELAY_MS = 500;
/** How long a message (and its Undo button) stays up. */
const MESSAGE_MS = 15_000;
/** A start counts as finished once the page has stayed responsive this long after the first run. */
const BOOT_GRACE_MS = 2500;
const AUTO_KEY = 'playground:auto';

type Pane = 'code' | 'preview' | 'console';

function readAuto(): boolean {
    try {
        return window.localStorage.getItem(AUTO_KEY) !== 'off';
    } catch {
        return true;
    }
}

function writeAuto(on: boolean): void {
    try {
        window.localStorage.setItem(AUTO_KEY, on ? 'on' : 'off');
    } catch {
        // not remembered, still works
    }
}

/** The page: tabs, toolbar, panes. All of the wiring between the editor, the runner and the preview lives here. */
export class App {
    project: Project = clone(DEFAULT_TEMPLATE.project);
    active = HTML_FILE;
    autoRun = readAuto();
    /** The project came from a link and has not been run: nothing runs until the person says so. */
    fromLink = false;

    private editor!: Editor;
    private preview!: Preview;
    private runner!: Runner;
    private output!: ConsolePanel;
    private runTimer: ReturnType<typeof setTimeout> | undefined;
    private saveTimer: ReturnType<typeof setTimeout> | undefined;
    private messageTimer: ReturnType<typeof setTimeout> | undefined;
    private previous: { project: Project; active: string } | null = null;
    private compilerNoticeShown = false;
    private readonly els = {
        tabs: $('tabs'),
        tabActions: $('tab-actions'),
        formHost: $('file-form-host'),
        status: $('status'),
        announce: $('announce'),
        message: $('message'),
        controls: $('controls'),
        notice: $('notice'),
        app: $('app'),
        panes: $('panes-nav'),
        splitter: $('splitter'),
        previewHost: $('preview-host'),
        consoleLog: $('console-log'),
        consoleTitle: $('console-count'),
    };

    constructor() {
        this.output = new ConsolePanel(this.els.consoleLog, (file, line, column) => this.jumpTo(file, line, column));
        this.preview = new Preview(
            this.els.previewHost,
            (event) => this.onSandbox(event),
            (network) => new URL(network ? 'sandbox-net.html' : 'sandbox.html', document.baseURI).href,
        );
        this.preview.onBlocked = (text) => {
            this.output.add({ level: 'system', group: 0, text });
            this.status(text);
        };
        this.runner = new Runner(this.preview, (project) => this.build(project));
        this.editor = new Editor($('editor-host'), {
            onChange: (name, text) => this.onEdit(name, text),
            onRun: () => void this.run(true),
        });
    }

    // ---------------------------------------------------------------- start up

    async start(): Promise<void> {
        const hash = location.hash;
        const shared = hash.startsWith('#p=') ? await decodeProject(hash) : null;
        if (shared) {
            // Whatever a link says, it starts with the network off: letting a link decide would let it open the
            // network before anyone has looked at its code. The person can turn it on themselves.
            this.project = { ...shared, network: false };
            this.fromLink = true;
        } else {
            const saved = loadSaved();
            if (saved) {
                this.project = saved.project;
                this.active = saved.active;
            }
            if (hash.startsWith('#p=')) this.say('That link could not be read, so the playground opened your last project instead.');
        }
        if (!this.project.files.some((f) => f.name === this.active)) this.active = this.project.files[0]!.name;

        this.buildControls();
        this.buildPanes();
        this.openFile(this.active);
        this.renderTabs();
        this.syncNotice();
        this.initSplitter();
        if (this.fromLink) {
            this.say('This project came from a link. It has not been run: press Run to run it.');
        } else if (didNotFinishBooting()) {
            // Last time the page froze, or was closed, before it had been running for a few seconds.
            clearBooting();
            this.say('The last run did not finish starting (the page may have hung), so the project was not run this time. Check the code, then press Run.');
        } else {
            markBooting();
            await this.run(true);
            setTimeout(clearBooting, BOOT_GRACE_MS); // a frozen page never reaches this
        }
    }

    // ---------------------------------------------------------------- building and running

    private async build(project: Project): Promise<BundleResult> {
        if (!entryOf(project)) return { ok: true, code: '', map: null, warnings: [], loops: 0 };
        if (!this.compilerNoticeShown) {
            this.compilerNoticeShown = true;
            this.status('Loading the compiler (about 11 MB, once)…');
        }
        // Everything the build needs loads here, on first use, so the editor itself opens quickly.
        const [esbuild, vendor, { bundleProject }] = await Promise.all([loadEsbuild(), import('virtual:vendor').then((m) => m.default), import('./bundle.ts')]);
        return bundleProject(project, esbuild, vendor);
    }

    private schedule(): void {
        clearTimeout(this.runTimer);
        if (!this.autoRun || this.fromLink) return;
        this.runTimer = setTimeout(() => void this.run(false), RUN_DELAY_MS);
    }

    /** Build and update the preview. `force` starts a fresh page even when nothing changed. */
    async run(force: boolean): Promise<void> {
        clearTimeout(this.runTimer);
        if (this.fromLink) this.leaveLinkMode();
        this.editor.clearDiagnostics();
        let outcome;
        try {
            outcome = await this.runner.update(this.project, { force });
        } catch (error) {
            this.status('The compiler could not be loaded. Check your connection and press Run.');
            this.say(`Could not build: ${error instanceof Error ? error.message : String(error)}`);
            return;
        }
        switch (outcome.kind) {
            case 'none':
            case 'stale':
                return;
            case 'css':
                this.status('Styles updated without reloading the page.');
                return;
            case 'failed': {
                this.showDiagnostics(outcome.errors);
                for (const d of outcome.errors) {
                    this.output.add({ level: 'error', group: 0, text: d.message, at: d.file ? { file: d.file, line: d.line, column: d.column } : null });
                }
                const n = outcome.errors.length;
                this.status(`Build failed: ${n} ${n === 1 ? 'error' : 'errors'}. The preview still shows the last good version.`);
                this.tell(`Build failed with ${n} ${n === 1 ? 'error' : 'errors'}.`);
                return;
            }
            case 'full':
                this.output.clear();
                this.showDiagnostics(outcome.warnings);
                for (const d of outcome.warnings) {
                    this.output.add({ level: 'warn', group: 0, text: d.message, at: d.file ? { file: d.file, line: d.line, column: d.column } : null });
                }
                this.status(`Ran in ${outcome.ms} ms.${outcome.loops ? ` ${outcome.loops} ${outcome.loops === 1 ? 'loop is' : 'loops are'} watched.` : ''}`);
        }
    }

    private showDiagnostics(list: Diagnostic[]): void {
        const byFile = new Map<string, Diagnostic[]>();
        for (const d of list) if (d.file) byFile.set(d.file, [...(byFile.get(d.file) ?? []), d]);
        for (const [file, diagnostics] of byFile) this.editor.setDiagnostics(file, diagnostics);
    }

    private onSandbox(event: SandboxEvent): void {
        const mapper = 'run' in event ? this.runner.mapperFor(event.run) : undefined;
        switch (event.t) {
            case 'log': {
                const at = event.stack && mapper ? mapper.firstOriginal(event.stack) : null;
                this.output.add({
                    level: event.level,
                    group: event.group,
                    text: event.text,
                    at,
                    ...(event.stack && mapper && event.level !== 'log' ? { stack: mapper.rewriteStack(event.stack) } : {}),
                });
                return;
            }
            case 'error': {
                const own = event.file === 'bundle.js' && mapper ? mapper.original(event.line, event.column) : null;
                const at = own ?? (mapper ? mapper.firstOriginal(event.stack) : null);
                this.output.add({
                    level: 'error',
                    group: 0,
                    text: `${event.name}: ${event.message}`,
                    at,
                    ...(event.stack && mapper ? { stack: mapper.rewriteStack(event.stack) } : {}),
                });
                if (at) this.editor.setDiagnostics(at.file, [{ file: at.file, line: at.line, column: at.column, length: 1, message: `${event.name}: ${event.message}`, severity: 'error' }]);
                return;
            }
            case 'csp':
                this.output.add({ level: 'system', group: 0, text: explainViolation(event.directive, event.blocked, this.project.network) });
                return;
            case 'clear':
                this.output.clear();
                return;
            default:
                return; // `ready` and `scroll` are handled by the preview itself
        }
    }

    // ---------------------------------------------------------------- editing

    private onEdit(name: string, text: string): void {
        const file = this.project.files.find((f) => f.name === name);
        if (!file || file.code === text) return;
        file.code = text;
        if (this.fromLink) return; // the shared project is not the person's yet; see leaveLinkMode
        this.afterChange();
    }

    private afterChange(): void {
        this.schedule();
        clearTimeout(this.saveTimer);
        this.saveTimer = setTimeout(() => this.persist(), SAVE_DELAY_MS);
    }

    private persist(): void {
        if (this.fromLink) return;
        if (!save(this.project, this.active)) this.status('Could not save to this browser, so your work is only kept while this tab is open.');
    }

    /** The person has started working on a project that arrived by link: it is theirs from now on. */
    private leaveLinkMode(): void {
        if (!this.fromLink) return;
        this.fromLink = false;
        history.replaceState(null, '', location.pathname + location.search);
        this.syncNotice();
        this.persist();
    }

    private openFile(name: string): void {
        const file = this.project.files.find((f) => f.name === name);
        if (!file) return;
        this.active = name;
        this.editor.open(name, file.code);
    }

    private jumpTo(file: string, line: number, column: number): void {
        if (!this.project.files.some((f) => f.name === file)) return;
        this.showPane('code');
        this.openFile(file);
        this.renderTabs();
        this.editor.reveal(line, column);
    }

    // ---------------------------------------------------------------- files

    private renderTabs(focusName?: string): void {
        const tabs = this.els.tabs;
        tabs.replaceChildren();
        for (const f of this.project.files) {
            const selected = f.name === this.active;
            const tab = h('button', { type: 'button', role: 'tab', class: 'tab', 'aria-selected': String(selected), tabindex: selected ? 0 : -1, 'data-name': f.name }, f.name);
            tab.addEventListener('click', () => {
                this.openFile(f.name);
                this.renderTabs();
                this.editor.focus();
            });
            tab.addEventListener('keydown', (e) => this.tabKeys(e, f.name));
            tabs.append(tab);
        }
        const actions = this.els.tabActions;
        actions.replaceChildren();
        const add = h('button', { type: 'button', class: 'mini', 'aria-label': 'New file' }, '+ New file');
        add.addEventListener('click', () => this.nameForm('add'));
        actions.append(add);
        if (this.active !== HTML_FILE) {
            const rename = h('button', { type: 'button', class: 'mini', 'aria-label': `Rename ${this.active}` }, 'Rename');
            rename.addEventListener('click', () => this.nameForm('rename'));
            const remove = h('button', { type: 'button', class: 'mini danger', 'aria-label': `Delete ${this.active}` }, 'Delete');
            remove.addEventListener('click', () => this.removeActive());
            actions.append(rename, remove);
        }
        if (focusName) tabs.querySelector<HTMLElement>(`[data-name="${CSS.escape(focusName)}"]`)?.focus();
    }

    private tabKeys(e: KeyboardEvent, name: string): void {
        const names = this.project.files.map((f) => f.name);
        const i = names.indexOf(name);
        const to = e.key === 'ArrowRight' ? names[(i + 1) % names.length] : e.key === 'ArrowLeft' ? names[(i - 1 + names.length) % names.length] : e.key === 'Home' ? names[0] : e.key === 'End' ? names.at(-1) : null;
        if (!to) return;
        e.preventDefault();
        this.openFile(to);
        this.renderTabs(to);
    }

    /** An inline form to name a new file or rename the open one. No dialogs: they are awkward to use and to test. */
    private nameForm(mode: 'add' | 'rename'): void {
        const initial = mode === 'add' ? freshName(this.project.files) : this.active;
        const input = h('input', { type: 'text', id: 'file-name', value: initial, maxlength: 40, autocomplete: 'off', spellcheck: 'false', 'aria-describedby': 'file-name-help' });
        const help = h('span', { id: 'file-name-help', class: 'help', role: 'alert' });
        const form = h('form', { class: 'name-form' }, h('label', { for: 'file-name' }, mode === 'add' ? 'New file name' : 'New name'), input, h('button', { type: 'submit', class: 'mini' }, mode === 'add' ? 'Add' : 'Rename'), h('button', { type: 'button', class: 'mini cancel' }, 'Cancel'), help);
        const close = () => {
            form.remove();
            this.renderTabs(this.active);
        };
        form.querySelector('.cancel')!.addEventListener('click', close);
        form.addEventListener('keydown', (e) => e.key === 'Escape' && close());
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            const name = input.value.trim();
            const problem = this.nameChangeProblem(mode, name);
            if (problem) {
                help.textContent = problem;
                input.setAttribute('aria-invalid', 'true');
                return;
            }
            if (mode === 'add') this.addFile(name);
            else this.renameActive(name);
        });
        this.els.formHost.replaceChildren(form);
        input.focus();
        input.select();
    }

    private nameChangeProblem(mode: 'add' | 'rename', name: string): string | null {
        const own = nameProblem(name);
        if (own) return own;
        const others = mode === 'rename' ? this.project.files.filter((f) => f.name !== this.active) : this.project.files;
        if (mode === 'rename' && name === this.active) return 'That is already its name.';
        return filesProblem([...others, { name, code: '' }]);
    }

    private addFile(name: string): void {
        const starter = extensionOf(name) === 'css' ? '/* styles */\n' : extensionOf(name) === 'json' ? '{}\n' : '';
        this.project.files.push({ name, code: starter });
        this.openFile(name);
        this.renderTabs();
        this.editor.focus();
        this.afterChange();
    }

    private renameActive(name: string): void {
        const old = this.active;
        const file = this.project.files.find((f) => f.name === old)!;
        file.name = name;
        this.editor.rename(old, name);
        this.active = name;
        this.renderTabs(name);
        this.afterChange();
    }

    private removeActive(): void {
        const name = this.active;
        if (name === HTML_FILE) return;
        const index = this.project.files.findIndex((f) => f.name === name);
        this.previous = { project: clone(this.project), active: name };
        this.project.files.splice(index, 1);
        this.editor.forget(name);
        this.openFile(this.project.files[Math.min(index, this.project.files.length - 1)]!.name);
        this.renderTabs();
        this.afterChange();
        this.offerUndo(`Deleted ${name}.`);
    }

    // ---------------------------------------------------------------- controls

    private buildControls(): void {
        const controls = this.els.controls;

        const chooser = h('select', { id: 'template', 'aria-label': 'Start from a template' }, h('option', { value: '' }, 'Start from…'), ...TEMPLATES.map((t) => h('option', { value: t.id }, t.title)));
        chooser.addEventListener('change', () => {
            const t = TEMPLATES.find((x) => x.id === chooser.value);
            chooser.value = '';
            if (t) this.loadProject(clone(t.project), `Loaded “${t.title}”.`);
        });

        const runButton = h('button', { type: 'button', id: 'run', class: 'primary', title: 'Run (Ctrl+Enter)', 'aria-keyshortcuts': 'Control+Enter' }, 'Run');
        runButton.addEventListener('click', () => void this.run(true));

        const auto = h('input', { type: 'checkbox', id: 'auto' });
        auto.checked = this.autoRun;
        auto.addEventListener('change', () => {
            this.autoRun = auto.checked;
            writeAuto(this.autoRun);
            if (this.autoRun) this.schedule();
        });

        const network = h('input', { type: 'checkbox', id: 'network', 'aria-describedby': 'network-help' });
        network.checked = this.project.network;
        network.addEventListener('change', () => {
            this.project.network = network.checked;
            this.syncNotice();
            if (this.fromLink) return; // a shared project does not run until Run is pressed, whatever else is changed
            this.afterChange();
            void this.run(true);
        });

        const share = h('button', { type: 'button', id: 'share', title: 'Copy a link that contains this whole project' }, 'Share');
        share.addEventListener('click', () => void this.share());

        const reset = h('button', { type: 'button', id: 'reset', title: 'Replace the project with the first template' }, 'Reset');
        reset.addEventListener('click', () => this.loadProject(clone(DEFAULT_TEMPLATE.project), 'Project reset.'));

        controls.replaceChildren(
            chooser,
            runButton,
            h('label', { class: 'check' }, auto, h('span', {}, 'Auto-run')),
            h('label', { class: 'check' }, network, h('span', {}, 'Allow network')),
            share,
            reset,
        );
    }

    private syncNotice(): void {
        const network = $<HTMLInputElement>('network');
        network.checked = this.project.network;
        const notice = this.els.notice;
        notice.replaceChildren();
        if (this.fromLink) {
            const run = h('button', { type: 'button', class: 'mini', id: 'run-shared' }, 'Run it');
            run.addEventListener('click', () => void this.run(true));
            const discard = h('button', { type: 'button', class: 'mini', id: 'discard-shared' }, 'Discard');
            discard.addEventListener('click', () => this.discardLink());
            notice.append(h('span', {}, 'This project came from a link. Look through the code before running it: it runs in a sandbox, but it is still someone else’s code. '), run, discard);
            notice.hidden = false;
        } else if (this.project.network) {
            notice.append(h('span', { id: 'network-help' }, 'Network is on: the preview can load https images, fonts and data, and import packages from esm.sh. Leave it off for code you do not trust.'));
            notice.hidden = false;
        } else {
            notice.hidden = true;
        }
    }

    private discardLink(): void {
        history.replaceState(null, '', location.pathname + location.search);
        const saved = loadSaved();
        this.fromLink = false;
        this.loadProject(saved ? saved.project : clone(DEFAULT_TEMPLATE.project), 'Discarded the shared project.', saved?.active, false);
    }

    /** Replace everything, keeping the old project for a moment so the person can take it back. */
    private loadProject(project: Project, message: string, active?: string, offerUndo = true): void {
        if (offerUndo) this.previous = { project: clone(this.project), active: this.active };
        for (const f of this.project.files) this.editor.forget(f.name);
        this.project = project;
        this.fromLink = false;
        history.replaceState(null, '', location.pathname + location.search);
        this.active = active && project.files.some((f) => f.name === active) ? active : (project.files.find((f) => isEntry(f.name))?.name ?? project.files[0]!.name);
        this.runner.invalidate();
        this.openFile(this.active);
        this.renderTabs();
        this.syncNotice();
        this.persist();
        void this.run(true);
        if (offerUndo) this.offerUndo(message);
        else this.say(message);
    }

    private offerUndo(message: string): void {
        const undo = h('button', { type: 'button', class: 'mini', id: 'undo-load' }, 'Undo');
        undo.addEventListener('click', () => {
            const before = this.previous;
            this.previous = null;
            if (before) this.loadProject(before.project, 'Restored.', before.active, false);
        });
        this.say(message, undo);
    }

    private async share(): Promise<void> {
        const fragment = await encodeProject(this.project);
        const url = `${location.origin}${location.pathname}#${fragment}`;
        history.replaceState(null, '', `#${fragment}`);
        const size = (new Blob([url]).size / 1000).toFixed(1);
        try {
            await navigator.clipboard.writeText(url);
            this.say(`Link copied (${size} kB). It contains the whole project; nothing is stored on a server.`);
        } catch {
            this.say(`The link is now in the address bar (${size} kB): copy it from there.`);
        }
    }

    // ---------------------------------------------------------------- panes

    private buildPanes(): void {
        const names: [Pane, string][] = [['code', 'Code'], ['preview', 'Preview'], ['console', 'Console']];
        const nav = this.els.panes;
        nav.replaceChildren(
            ...names.map(([pane, label]) => {
                const b = h('button', { type: 'button', class: 'pane-tab', 'aria-pressed': String(pane === 'code'), 'data-pane': pane }, label);
                b.addEventListener('click', () => this.showPane(pane));
                return b;
            }),
        );
        $<HTMLSelectElement>('console-filter').addEventListener('change', (e) => this.output.setFilter((e.target as HTMLSelectElement).value as Filter));
        $('console-clear').addEventListener('click', () => this.output.clear());
        $('reload').addEventListener('click', () => void this.run(true));
        this.output.onCounts = (warnings, errors) => {
            this.els.consoleTitle.textContent = errors || warnings ? `(${errors} ${errors === 1 ? 'error' : 'errors'}, ${warnings} ${warnings === 1 ? 'warning' : 'warnings'})` : '';
        };
    }

    private showPane(pane: Pane): void {
        this.els.app.dataset.pane = pane;
        for (const b of this.els.panes.querySelectorAll<HTMLElement>('[data-pane]')) b.setAttribute('aria-pressed', String(b.dataset.pane === pane));
    }

    private initSplitter(): void {
        const { splitter, app } = this.els;
        let percent = 50;
        const set = (value: number) => {
            percent = Math.min(80, Math.max(20, value));
            app.style.setProperty('--split', `${percent}%`);
            splitter.setAttribute('aria-valuenow', String(Math.round(percent)));
        };
        set(50);
        splitter.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') set(percent - 3);
            else if (e.key === 'ArrowRight') set(percent + 3);
            else return;
            e.preventDefault();
        });
        splitter.addEventListener('pointerdown', (e) => {
            splitter.setPointerCapture(e.pointerId);
            const rect = app.getBoundingClientRect();
            const move = (m: PointerEvent) => set(((m.clientX - rect.left) / rect.width) * 100);
            const stop = () => {
                splitter.removeEventListener('pointermove', move);
                splitter.removeEventListener('pointerup', stop);
            };
            splitter.addEventListener('pointermove', move);
            splitter.addEventListener('pointerup', stop);
        });
    }

    // ---------------------------------------------------------------- messages to the person

    private status(text: string): void {
        this.els.status.textContent = text;
    }

    /** Show a message, and announce it to assistive technology (the quiet status line is not announced). */
    private say(text: string, action?: HTMLElement): void {
        const { message } = this.els;
        message.replaceChildren(text, ...(action ? [action] : []));
        message.hidden = false;
        clearTimeout(this.messageTimer);
        this.messageTimer = setTimeout(() => (message.hidden = true), MESSAGE_MS);
        this.tell(text);
    }

    /** Announce to assistive technology only. */
    private tell(text: string): void {
        this.els.announce.textContent = '';
        setTimeout(() => (this.els.announce.textContent = text), 40);
    }

    get previewHandle(): Preview {
        return this.preview;
    }
}

/** What the preview's content security policy blocked, in words a person can act on. */
export function explainViolation(directive: string, blocked: string, network: boolean): string {
    if (directive.startsWith('form-action')) {
        return 'A form tried to submit. The preview cannot navigate or send data anywhere, so handle the form\'s submit event and call event.preventDefault().';
    }
    if (directive.startsWith('script-src') && (blocked === 'inline' || blocked === '')) {
        return 'Blocked an inline script or event-handler attribute (such as onclick="…"). The preview only runs code from your script files: put it in main.js and use addEventListener.';
    }
    const what = blocked && blocked !== 'inline' ? blocked : 'a resource';
    const hint = network ? 'Only https resources (and packages from https://esm.sh) are allowed.' : 'Network is off. Turn on “Allow network” to load https images, fonts and data.';
    return `Blocked ${what} (${directive}). ${hint}`;
}
