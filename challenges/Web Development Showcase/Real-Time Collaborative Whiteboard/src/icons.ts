import type { ToolId } from './tools.ts';

/** 24x24 line icons, drawn for this project. `currentColor` so they follow the button's text colour. */
const PATHS: Record<string, string> = {
    select: '<path d="M5 3l14 8-6 2-2 6z"/>',
    hand: '<path d="M8 12V5a1.5 1.5 0 013 0v6m0-1V4a1.5 1.5 0 013 0v7m0-5a1.5 1.5 0 013 0v7a7 7 0 01-7 7h-1a6 6 0 01-5-3l-2.5-4a1.5 1.5 0 012.4-1.6L8 14"/>',
    pen: '<path d="M4 20l1-4L16 5a2.1 2.1 0 013 3L8 19z"/><path d="M14 7l3 3"/>',
    highlighter: '<path d="M9 14l6-6 3 3-6 6H9z"/><path d="M9 17l-3 3h5M15 8l2-2 3 3-2 2"/>',
    eraser: '<path d="M8 20h12M6 15l8-9a2 2 0 013 0l2 2a2 2 0 010 3l-6 7H9a2 2 0 01-1.5-.6z"/>',
    line: '<path d="M5 19L19 5"/>',
    arrow: '<path d="M5 19L19 5M10 5h9v9"/>',
    rect: '<rect x="4" y="6" width="16" height="12" rx="1.5"/>',
    ellipse: '<ellipse cx="12" cy="12" rx="8" ry="6"/>',
    text: '<path d="M6 6h12M12 6v13M9 19h6"/>',
    sticky: '<path d="M5 4h14v10l-5 6H5z"/><path d="M14 20v-5a1 1 0 011-1h4"/>',
    undo: '<path d="M9 14L4 9l5-5"/><path d="M4 9h10a6 6 0 010 12h-3"/>',
    redo: '<path d="M15 14l5-5-5-5"/><path d="M20 9H10a6 6 0 000 12h3"/>',
    plus: '<path d="M12 5v14M5 12h14"/>',
    minus: '<path d="M5 12h14"/>',
    fit: '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
    download: '<path d="M12 4v11m0 0l-4-4m4 4l4-4M5 20h14"/>',
    link: '<path d="M10 14a4 4 0 005.7 0l3-3a4 4 0 00-5.7-5.7l-1 1M14 10a4 4 0 00-5.7 0l-3 3a4 4 0 005.7 5.7l1-1"/>',
    users: '<circle cx="9" cy="8" r="3.5"/><path d="M2.5 20a6.5 6.5 0 0113 0M16 4.5a3.5 3.5 0 010 7M18 14a6.5 6.5 0 013.5 6"/>',
};

export function icon(name: ToolId | keyof typeof PATHS): string {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">${PATHS[name] ?? ''}</svg>`;
}
