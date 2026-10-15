type Attrs = Record<string, string | boolean | number | undefined>;

/**
 * A tiny element builder. Text goes in as text nodes, never as HTML, so a file name, a console
 * message or an error from the preview can never inject markup into the playground.
 */
export function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Attrs = {}, ...children: (Node | string)[]): HTMLElementTagNameMap[K] {
    const node = document.createElement(tag);
    for (const [name, value] of Object.entries(attrs)) {
        if (value === undefined || value === false) continue;
        node.setAttribute(name, value === true ? '' : String(value));
    }
    node.append(...children);
    return node;
}

export function $<T extends HTMLElement>(id: string): T {
    const el = document.getElementById(id);
    if (!el) throw new Error(`#${id} is missing from the page`);
    return el as T;
}
