import { highlightRuns } from './fuzzy.js';
import { createSearch } from './search.js';

/**
 * The palette UI: a modal `<dialog>` holding an ARIA combobox (the input) that controls a listbox
 * of grouped options. Focus never leaves the input; the highlighted option is announced through
 * `aria-activedescendant`. The browser supplies the focus trap, the inert page behind it and Esc.
 *
 * Dependencies come in as options so the whole thing runs in jsdom and in tests without a page:
 *   loadItems()   -> Promise<item[]> (the built index; actions are always available without it)
 *   actions       -> action items with `run(context)`
 *   context       -> passed to `run`
 *   recents       -> { get, push, clear }
 *   navigate(url) / openTab(url) -> how a page or heading is opened
 *   currentPage   -> 'getting-started.html', to mark the page you are already on
 */

const KIND_LABEL = { page: 'Page', heading: 'Heading', action: 'Action', recent: 'Recent' };
const KIND_GLYPH = { page: '▤', heading: '#', action: '⚡' };
const STATUS_DELAY = 250;

function el(doc, tag, className, text) {
    const node = doc.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
}

/** Append `text` to `parent`, with the matched characters wrapped in `<span class="cp-m">`. */
export function appendHighlighted(doc, parent, text, positions) {
    for (const run of highlightRuns(text, positions)) {
        if (run.match) parent.append(el(doc, 'span', 'cp-m', run.text));
        else parent.append(doc.createTextNode(run.text));
    }
}

export function createPalette({
    doc = document,
    loadItems,
    actions = [],
    context = {},
    recents,
    navigate = (url) => doc.defaultView.location.assign(url),
    openTab = (url) => doc.defaultView.open(url, '_blank', 'noopener'),
    currentPage = '',
}) {
    let dialog;
    let input;
    let listbox;
    let empty;
    let status;
    let search = createSearch(actions);
    let loaded = 'idle'; // idle | loading | ready | failed
    let opener = null;
    let flat = []; // the rendered options, in order: [{ item, node }]
    let active = -1;
    let statusTimer;

    const actionById = new Map(actions.map((action) => [action.id, action]));

    function build() {
        dialog = el(doc, 'dialog', 'cp');
        dialog.setAttribute('aria-label', 'Command palette');

        const head = el(doc, 'div', 'cp-head');
        input = el(doc, 'input', 'cp-input');
        input.type = 'text';
        input.id = 'cp-input';
        input.placeholder = 'Search pages, headings and actions';
        input.setAttribute('role', 'combobox');
        input.setAttribute('aria-label', 'Search pages, headings and actions');
        input.setAttribute('aria-controls', 'cp-list');
        input.setAttribute('aria-expanded', 'true');
        input.setAttribute('aria-autocomplete', 'list');
        input.setAttribute('aria-haspopup', 'listbox');
        input.setAttribute('autocomplete', 'off');
        input.setAttribute('autocapitalize', 'off');
        input.setAttribute('autocorrect', 'off');
        input.setAttribute('spellcheck', 'false');
        input.setAttribute('enterkeyhint', 'go');
        const close = el(doc, 'button', 'cp-close', 'Close');
        close.type = 'button';
        close.addEventListener('click', () => api.close());
        head.append(input, close);

        listbox = el(doc, 'div', 'cp-list');
        listbox.id = 'cp-list';
        listbox.setAttribute('role', 'listbox');
        listbox.setAttribute('aria-label', 'Results');
        empty = el(doc, 'p', 'cp-empty');
        empty.hidden = true;
        status = el(doc, 'p', 'cp-status');
        status.setAttribute('role', 'status');

        const foot = el(doc, 'p', 'cp-foot');
        foot.setAttribute('aria-hidden', 'true');
        foot.append(
            ...[
                ['↑↓', 'move'],
                ['↵', 'open'],
                ['Ctrl ↵', 'new tab'],
                ['Esc', 'close'],
            ].flatMap(([keys, label]) => [el(doc, 'kbd', '', keys), doc.createTextNode(` ${label} `)]),
        );

        dialog.append(head, listbox, empty, status, foot);
        dialog.dataset.index = loaded;
        doc.body.append(dialog);

        input.addEventListener('input', () => render(input.value));
        input.addEventListener('keydown', onKeydown);
        // Keep focus in the input when the list is clicked or scrolled with the mouse.
        listbox.addEventListener('mousedown', (event) => event.preventDefault());
        listbox.addEventListener('click', onClick);
        listbox.addEventListener('mousemove', (event) => {
            const option = event.target.closest?.('[role="option"]');
            const index = option ? flat.findIndex((entry) => entry.node === option) : -1;
            if (index >= 0 && index !== active) setActive(index, false);
        });
        // A click on the backdrop lands on the dialog element itself (it has no padding).
        dialog.addEventListener('click', (event) => {
            if (event.target === dialog) api.close();
        });
        dialog.addEventListener('close', onClosed);
        // The browser keeps focus off the page behind a modal dialog, but lets Tab leave for its own
        // chrome; wrap between the two controls so Tab and Shift+Tab stay inside.
        dialog.addEventListener('keydown', (event) => {
            if (event.key !== 'Tab') return;
            const [first, last] = [input, close];
            if (event.shiftKey && doc.activeElement === first) handled(event, () => last.focus());
            else if (!event.shiftKey && doc.activeElement === last) handled(event, () => first.focus());
        });
    }

    /** The index state is mirrored on the dialog (`data-index`) so tests and styles can see it. */
    function setLoaded(state) {
        loaded = state;
        if (dialog) dialog.dataset.index = state;
    }

    async function ensureItems() {
        if (loaded !== 'idle') return;
        setLoaded('loading');
        try {
            const items = await loadItems();
            search = createSearch([...actions, ...items]);
            setLoaded('ready');
        } catch {
            setLoaded('failed');
        }
        if (dialog?.open) render(input.value);
    }

    function render(query) {
        const sections = search.search(query, { recentIds: recents?.get() ?? [] });
        listbox.replaceChildren();
        flat = [];
        for (const section of sections) {
            const group = el(doc, 'div', 'cp-group');
            group.setAttribute('role', 'group');
            const label = el(doc, 'div', 'cp-group-label', section.label);
            label.id = `cp-group-${section.type}`;
            label.setAttribute('role', 'presentation');
            group.setAttribute('aria-labelledby', label.id);
            group.append(label);
            for (const { item, positions } of section.results) {
                const node = optionNode(item, positions, section.type === 'recent');
                node.id = `cp-opt-${flat.length}`;
                flat.push({ item, node });
                group.append(node);
            }
            listbox.append(group);
        }
        const has = flat.length > 0;
        listbox.hidden = !has;
        empty.hidden = has;
        input.setAttribute('aria-expanded', String(has));
        if (!has) {
            const note = loaded === 'failed' ? ' The page index could not be loaded, so only actions are searched.' : '';
            empty.textContent = query.trim()
                ? `No results for “${query.trim()}”. Try fewer or different letters.${note}`
                : `Nothing to show yet.${note}`;
        }
        setActive(has ? 0 : -1, false);
        announce(has ? `${flat.length} result${flat.length === 1 ? '' : 's'}` : 'No results');
    }

    function optionNode(item, positions, isRecent) {
        const node = el(doc, 'div', 'cp-option');
        node.setAttribute('role', 'option');
        node.setAttribute('aria-selected', 'false');
        node.dataset.id = item.id;
        const glyph = el(doc, 'span', 'cp-glyph', KIND_GLYPH[item.type] ?? '·');
        glyph.setAttribute('aria-hidden', 'true');
        node.append(glyph);
        const main = el(doc, 'span', 'cp-main');
        const title = el(doc, 'span', 'cp-title');
        appendHighlighted(doc, title, item.title, positions);
        main.append(title);
        const sub = item.hint ?? item.description;
        if (sub) main.append(el(doc, 'span', 'cp-hint', sub));
        node.append(main);
        if (item.url && urlPath(item.url) === currentPage && !item.url.includes('#')) {
            node.append(el(doc, 'span', 'cp-badge', 'You are here'));
        } else if (isRecent) {
            node.append(el(doc, 'span', 'cp-kind', KIND_LABEL[item.type]));
        }
        return node;
    }

    function announce(message) {
        clearTimeout(statusTimer);
        statusTimer = setTimeout(() => {
            if (status) status.textContent = message;
        }, STATUS_DELAY);
    }

    function setActive(index, scroll = true) {
        if (active >= 0 && flat[active]) {
            flat[active].node.setAttribute('aria-selected', 'false');
        }
        active = index;
        if (index < 0 || !flat[index]) {
            input.removeAttribute('aria-activedescendant');
            return;
        }
        const { node } = flat[index];
        node.setAttribute('aria-selected', 'true');
        input.setAttribute('aria-activedescendant', node.id);
        if (scroll) node.scrollIntoView?.({ block: 'nearest' });
    }

    function move(step) {
        if (!flat.length) return;
        setActive((active + step + flat.length) % flat.length);
    }

    function onKeydown(event) {
        // An IME is composing: Enter and the arrows belong to it, not to us.
        if (event.isComposing || event.keyCode === 229) return;
        const ctrl = event.ctrlKey || event.metaKey;
        switch (event.key) {
            case 'ArrowDown':
                return handled(event, () => move(1));
            case 'ArrowUp':
                return handled(event, () => move(-1));
            case 'n':
            case 'j':
                if (event.ctrlKey && !event.altKey) return handled(event, () => move(1));
                return;
            case 'p':
            case 'k':
                if (event.ctrlKey && !event.altKey) return handled(event, () => move(-1));
                return;
            case 'PageDown':
                return handled(event, () => setActive(Math.min(active + 5, flat.length - 1)));
            case 'PageUp':
                return handled(event, () => setActive(Math.max(active - 5, 0)));
            case 'Home':
                // Home/End keep their caret job while there is text; with an empty box or Ctrl they move the list.
                if (ctrl || !input.value) return handled(event, () => setActive(flat.length ? 0 : -1));
                return;
            case 'End':
                if (ctrl || !input.value) return handled(event, () => setActive(flat.length - 1));
                return;
            case 'Enter':
                return handled(event, () => choose(active, event.ctrlKey || event.metaKey || event.shiftKey));
            default:
        }
    }

    function handled(event, fn) {
        event.preventDefault();
        fn();
    }

    function onClick(event) {
        const option = event.target.closest?.('[role="option"]');
        const index = option ? flat.findIndex((entry) => entry.node === option) : -1;
        if (index >= 0) choose(index, event.ctrlKey || event.metaKey || event.shiftKey);
    }

    function choose(index, newTab) {
        const entry = flat[index];
        if (!entry) return;
        const { item } = entry;
        recents?.push(item.id);
        if (item.type === 'action') {
            api.close();
            return actionById.get(item.id)?.run(context);
        }
        if (newTab) return openTab(item.url);
        api.close();
        return navigate(item.url);
    }

    function onClosed() {
        const target = opener;
        opener = null;
        if (target?.isConnected && typeof target.focus === 'function') target.focus();
    }

    const api = {
        get isOpen() {
            return Boolean(dialog?.open);
        },
        open() {
            if (!dialog) build();
            if (dialog.open) return;
            opener = doc.activeElement;
            input.value = '';
            ensureItems();
            render('');
            dialog.showModal();
            input.focus();
        },
        close() {
            if (dialog?.open) dialog.close();
        },
        toggle() {
            if (api.isOpen) api.close();
            else api.open();
        },
        /** Start loading the index before the first open (idle time, hover, focus). */
        preload() {
            return ensureItems();
        },
        get element() {
            return dialog;
        },
    };
    return api;
}

/** `getting-started.html#install` -> `getting-started.html`; the bare site root is `index.html`. */
export function urlPath(url) {
    const path = url.split('#')[0].split('?')[0].split('/').pop();
    return path || 'index.html';
}
