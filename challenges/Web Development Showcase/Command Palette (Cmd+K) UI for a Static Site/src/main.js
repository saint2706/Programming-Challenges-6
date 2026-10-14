import './palette.css';
import { ACTIONS } from './actions.js';
import { createPalette, urlPath } from './palette.js';
import { createRecents } from './recent.js';
import { createTheme } from './theme.js';
import { createToast } from './toast.js';

const win = window;
const currentPage = urlPath(win.location.pathname);

const theme = createTheme();
const recents = createRecents();
const toast = createToast();

/** Open a page or heading. A heading on this very page is a hash jump, not a reload. */
function navigate(url) {
    const [, hash] = url.split('#');
    if (urlPath(url) !== currentPage) return win.location.assign(url);
    if (!hash) return win.scrollTo({ top: 0 });
    win.location.hash = hash;
    const target = document.getElementById(hash);
    if (target) {
        target.setAttribute('tabindex', '-1');
        target.focus({ preventScroll: true });
    }
}

const palette = createPalette({
    // A separate chunk: pages that never open the palette never download the index.
    loadItems: async () => (await import('./generated/search-index.json')).default.items,
    actions: ACTIONS,
    context: { theme, recents, toast, win },
    recents,
    navigate,
    currentPage,
});

function isTyping(target) {
    return (
        target instanceof win.HTMLElement &&
        (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName))
    );
}

document.addEventListener('keydown', (event) => {
    if (event.isComposing) return;
    const mod = event.ctrlKey || event.metaKey;
    if (mod && !event.altKey && !event.shiftKey && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        palette.toggle();
    } else if (event.key === '/' && !mod && !event.altKey && !palette.isOpen && !isTyping(event.target)) {
        event.preventDefault();
        palette.open();
    }
});

// The trigger buttons are `hidden` in the HTML and only appear when this script runs, so nobody
// gets a button that does nothing.
const isApple = /Mac|iPhone|iPad/.test(win.navigator.userAgentData?.platform ?? win.navigator.userAgent);
for (const button of document.querySelectorAll('[data-palette-open]')) {
    button.hidden = false;
    for (const hint of button.querySelectorAll('[data-palette-hint]')) hint.textContent = isApple ? '⌘K' : 'Ctrl K';
    button.addEventListener('click', () => palette.open());
    button.addEventListener('pointerenter', () => palette.preload(), { once: true });
    button.addEventListener('focus', () => palette.preload(), { once: true });
}
(win.requestIdleCallback ?? ((fn) => setTimeout(fn, 1500)))(() => palette.preload());

// Test hook, same convention as the other challenges: `?e2e` exposes the palette to the browser tests.
if (new URLSearchParams(win.location.search).has('e2e')) win.__palette = palette;
