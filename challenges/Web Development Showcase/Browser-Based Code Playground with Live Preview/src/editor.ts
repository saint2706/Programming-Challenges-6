import { indentWithTab, toggleTabFocusMode } from '@codemirror/commands';
import { css } from '@codemirror/lang-css';
import { html } from '@codemirror/lang-html';
import { javascript } from '@codemirror/lang-javascript';
import { HighlightStyle, syntaxHighlighting } from '@codemirror/language';
import { type Diagnostic as CmDiagnostic, lintGutter, setDiagnostics } from '@codemirror/lint';
import { EditorState, type Extension } from '@codemirror/state';
import { EditorView, keymap } from '@codemirror/view';
import { tags as t } from '@lezer/highlight';
import { basicSetup } from 'codemirror';
import type { Diagnostic } from './bundle.ts';
import { extensionOf } from './project.ts';

export interface EditorHooks {
    onChange(name: string, text: string): void;
    /** Mod-Enter inside the editor. */
    onRun(): void;
}

/** Colours come from CSS variables, so light and dark mode need no JavaScript (see style.css). */
const highlight = HighlightStyle.define([
    { tag: [t.keyword, t.modifier, t.operatorKeyword, t.controlKeyword], color: 'var(--tok-keyword)' },
    { tag: [t.string, t.special(t.string), t.regexp], color: 'var(--tok-string)' },
    { tag: [t.number, t.bool, t.null, t.atom, t.constant(t.name)], color: 'var(--tok-number)' },
    { tag: [t.comment, t.lineComment, t.blockComment], color: 'var(--tok-comment)', fontStyle: 'italic' },
    { tag: [t.function(t.variableName), t.function(t.propertyName), t.definition(t.function(t.variableName))], color: 'var(--tok-function)' },
    { tag: [t.tagName, t.angleBracket], color: 'var(--tok-tag)' },
    { tag: [t.attributeName, t.propertyName], color: 'var(--tok-attr)' },
    { tag: [t.typeName, t.className, t.namespace], color: 'var(--tok-type)' },
    { tag: [t.variableName, t.definition(t.variableName)], color: 'var(--tok-variable)' },
    { tag: t.invalid, color: 'var(--tok-invalid)' },
]);

const theme = EditorView.theme({
    '&': { height: '100%', backgroundColor: 'var(--editor-bg)', color: 'var(--editor-fg)', fontSize: '14px' },
    '.cm-scroller': { fontFamily: 'ui-monospace, "Cascadia Code", "SF Mono", Consolas, monospace', lineHeight: '1.5' },
    '.cm-content': { caretColor: 'var(--editor-fg)' },
    '.cm-gutters': { backgroundColor: 'var(--editor-gutter)', color: 'var(--editor-gutter-fg)', border: 'none' },
    '.cm-activeLine, .cm-activeLineGutter': { backgroundColor: 'var(--editor-active)' },
    '&.cm-focused .cm-selectionBackground, .cm-selectionBackground, ::selection': { backgroundColor: 'var(--editor-selection)' },
    '&.cm-focused': { outline: '2px solid var(--focus)', outlineOffset: '-2px' },
    '.cm-tooltip': { backgroundColor: 'var(--panel)', color: 'var(--ink)', border: '1px solid var(--line)' },
});

function language(name: string): Extension {
    switch (extensionOf(name)) {
        case 'html':
            return html();
        case 'css':
            return css();
        case 'ts':
            return javascript({ typescript: true });
        case 'tsx':
            return javascript({ typescript: true, jsx: true });
        case 'jsx':
            return javascript({ jsx: true });
        default:
            return javascript();
    }
}

/** Where a 1-based line and column are in `state`, kept inside the document. */
export function positionOf(state: EditorState, line: number, column: number): number {
    const l = state.doc.line(Math.min(Math.max(1, line), state.doc.lines));
    return Math.min(l.to, l.from + Math.max(0, column - 1));
}

/**
 * One CodeMirror view that shows whichever file is open. Each file keeps its own state, so undo
 * history, selection and scroll position survive switching tabs.
 */
export class Editor {
    readonly view: EditorView;
    private readonly hooks: EditorHooks;
    private readonly states = new Map<string, EditorState>();
    private readonly diagnostics = new Map<string, Diagnostic[]>();
    private current = '';

    constructor(parent: HTMLElement, hooks: EditorHooks) {
        this.hooks = hooks;
        this.view = new EditorView({ parent, state: EditorState.create({ doc: '', extensions: [] }) });
    }

    get active(): string {
        return this.current;
    }

    private makeState(name: string, doc: string): EditorState {
        return EditorState.create({
            doc,
            extensions: [
                basicSetup,
                language(name),
                syntaxHighlighting(highlight),
                theme,
                lintGutter(),
                EditorView.lineWrapping,
                EditorView.contentAttributes.of({ 'aria-label': `Code editor: ${name}`, spellcheck: 'false', autocapitalize: 'off', autocorrect: 'off' }),
                keymap.of([
                    { key: 'Mod-Enter', run: () => (this.hooks.onRun(), true), preventDefault: true },
                    // Tab indents; Ctrl-M (or Escape, then Tab) hands Tab back to the page so a keyboard user can leave.
                    { key: 'Ctrl-m', mac: 'Shift-Alt-m', run: toggleTabFocusMode },
                    indentWithTab,
                ]),
                EditorView.updateListener.of((update) => {
                    if (update.docChanged && this.current === name) this.hooks.onChange(name, update.state.doc.toString());
                }),
            ],
        });
    }

    /** Show `name`, creating its state from `code` the first time. */
    open(name: string, code: string): void {
        if (this.current && this.current !== name) this.states.set(this.current, this.view.state);
        this.current = name;
        let state = this.states.get(name);
        if (!state) {
            state = this.makeState(name, code);
            this.states.set(name, state);
        }
        this.view.setState(state);
        this.applyDiagnostics();
    }

    /** What a file currently says, whether or not it is the one on screen. */
    text(name: string): string | undefined {
        if (name === this.current) return this.view.state.doc.toString();
        return this.states.get(name)?.doc.toString();
    }

    /** Replace a file's text from outside (loading a project, switching template). Forgets its history. */
    reset(name: string, code: string): void {
        this.states.delete(name);
        if (name === this.current) this.open(name, code);
    }

    forget(name: string): void {
        this.states.delete(name);
        this.diagnostics.delete(name);
        if (this.current === name) this.current = '';
    }

    rename(from: string, to: string): void {
        const state = from === this.current ? this.view.state : this.states.get(from);
        const diags = this.diagnostics.get(from);
        const code = state?.doc.toString() ?? '';
        this.states.delete(from);
        this.diagnostics.delete(from);
        if (diags) this.diagnostics.set(to, diags);
        // The language and the accessible name depend on the file name, so the state is rebuilt (history is kept out of it).
        const wasOpen = this.current === from;
        if (wasOpen) this.current = '';
        this.states.set(to, this.makeState(to, code));
        if (wasOpen) this.open(to, code);
    }

    /** Problems to underline in `name`. Replaces whatever was shown for that file. */
    setDiagnostics(name: string, diagnostics: Diagnostic[]): void {
        if (diagnostics.length) this.diagnostics.set(name, diagnostics);
        else this.diagnostics.delete(name);
        if (name === this.current) this.applyDiagnostics();
    }

    clearDiagnostics(): void {
        this.diagnostics.clear();
        this.applyDiagnostics();
    }

    private applyDiagnostics(): void {
        const found = this.diagnostics.get(this.current) ?? [];
        const state = this.view.state;
        const converted: CmDiagnostic[] = found.map((d) => {
            const from = positionOf(state, d.line, d.column);
            return { from, to: Math.min(state.doc.length, Math.max(from + 1, from + d.length)), severity: d.severity, message: d.message, source: 'playground' };
        });
        this.view.dispatch(setDiagnostics(state, converted));
    }

    reveal(line: number, column: number): void {
        const pos = positionOf(this.view.state, line, column);
        this.view.dispatch({ selection: { anchor: pos }, effects: EditorView.scrollIntoView(pos, { y: 'center' }) });
        this.view.focus();
    }

    focus(): void {
        this.view.focus();
    }
}
