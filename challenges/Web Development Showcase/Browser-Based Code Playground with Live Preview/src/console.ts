import { h } from './dom.ts';
import type { Level } from './messages.ts';

export interface Entry {
    level: Level;
    text: string;
    /** Indentation from console.group, 0 to 10. */
    group: number;
    /** Where it came from in the person's files, when known. */
    at?: { file: string; line: number; column: number } | null;
    /** A mapped stack trace, shown on demand. */
    stack?: string;
}

export type Filter = 'all' | 'warn' | 'error';

/** Entries waiting to be drawn. A page that posts faster than a frame can draw has the oldest dropped. */
const MAX_QUEUE = 2000;

/** Entries kept on screen. The sandbox already stops a run at 1,000 messages; this covers many runs. */
const MAX_ENTRIES = 1000;

/**
 * The console pane. Every string is set as text, an entry is a few small elements, and drawing is
 * batched to one pass per frame, so a run that logs thousands of lines cannot make the page crawl.
 */
export class ConsolePanel {
    readonly el: HTMLElement;
    private readonly onJump: (file: string, line: number, column: number) => void;
    private queue: Entry[] = [];
    private frame = 0;
    private warnings = 0;
    private errors = 0;
    onCounts: (warnings: number, errors: number) => void = () => {};

    constructor(el: HTMLElement, onJump: (file: string, line: number, column: number) => void) {
        this.el = el;
        this.onJump = onJump;
        el.dataset.filter = 'all';
    }

    add(entry: Entry): void {
        this.queue.push(entry);
        if (this.queue.length > MAX_QUEUE) this.queue.splice(0, this.queue.length - MAX_QUEUE);
        if (!this.frame) this.frame = requestAnimationFrame(() => this.flush());
    }

    clear(): void {
        this.queue = [];
        this.el.replaceChildren();
        this.warnings = this.errors = 0;
        this.onCounts(0, 0);
    }

    setFilter(filter: Filter): void {
        this.el.dataset.filter = filter;
    }

    get count(): number {
        return this.el.childElementCount;
    }

    private flush(): void {
        this.frame = 0;
        const atBottom = this.el.scrollHeight - this.el.scrollTop - this.el.clientHeight < 24;
        const fragment = document.createDocumentFragment();
        for (const entry of this.queue) {
            if (entry.level === 'warn') this.warnings++;
            if (entry.level === 'error') this.errors++;
            fragment.append(this.render(entry));
        }
        this.queue = [];
        this.el.append(fragment);
        while (this.el.childElementCount > MAX_ENTRIES) this.el.firstElementChild?.remove();
        if (atBottom) this.el.scrollTop = this.el.scrollHeight;
        this.onCounts(this.warnings, this.errors);
    }

    private render(entry: Entry): HTMLElement {
        const row = h('div', { class: `entry l-${entry.level}`, 'data-level': entry.level });
        row.style.setProperty('--g', String(entry.group));
        const label = LABELS[entry.level];
        if (label) row.append(h('span', { class: 'sr-only' }, `${label}: `));
        row.append(h('span', { class: 'msg' }, entry.text));
        if (entry.at) {
            const { file, line, column } = entry.at;
            const jump = h('button', { type: 'button', class: 'where', 'aria-label': `${file}:${line}:${column} (go to this line)` }, `${file}:${line}:${column}`);
            jump.addEventListener('click', () => this.onJump(file, line, column));
            row.append(jump);
        }
        if (entry.stack) {
            row.append(h('details', { class: 'stack' }, h('summary', {}, 'Stack'), h('pre', {}, entry.stack)));
        }
        return row;
    }
}

const LABELS: Partial<Record<Level, string>> = { error: 'Error', warn: 'Warning', system: 'Playground' };
