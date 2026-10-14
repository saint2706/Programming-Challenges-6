/**
 * The palette's built-in commands. Each is a search item (`type: 'action'`) plus `run(context)`
 * where `context = { theme, recents, toast, win }`. They live in code, not in the built page index,
 * because they are behaviour, not content.
 */

async function copyText(win, text) {
    try {
        await win.navigator.clipboard.writeText(text);
        return true;
    } catch {
        // No async clipboard (insecure context, permission): try the old route.
    }
    const doc = win.document;
    const field = doc.createElement('textarea');
    field.value = text;
    field.setAttribute('readonly', '');
    field.style.cssText = 'position:fixed;top:0;left:0;opacity:0';
    doc.body.append(field);
    field.select();
    let copied = false;
    try {
        copied = doc.execCommand('copy');
    } catch {
        // ignore: reported as a failure below
    }
    field.remove();
    return copied;
}

export const ACTIONS = [
    {
        id: 'a:theme',
        type: 'action',
        title: 'Toggle color theme',
        description: 'System, light or dark',
        keywords: ['dark', 'light', 'mode', 'appearance'],
        run: ({ theme, toast }) => toast(`Theme: ${theme.cycle()}`),
    },
    {
        id: 'a:copy-link',
        type: 'action',
        title: 'Copy link to this page',
        description: 'Puts the address on the clipboard',
        keywords: ['url', 'share', 'clipboard'],
        run: async ({ toast, win }) => {
            const ok = await copyText(win, win.location.href);
            toast(ok ? 'Link copied' : 'Could not copy the link');
        },
    },
    {
        id: 'a:top',
        type: 'action',
        title: 'Scroll to top',
        keywords: ['start', 'beginning'],
        run: ({ win }) => win.scrollTo({ top: 0 }),
    },
    {
        id: 'a:print',
        type: 'action',
        title: 'Print this page',
        keywords: ['pdf'],
        run: ({ win }) => win.print(),
    },
    {
        id: 'a:clear-recent',
        type: 'action',
        title: 'Clear recent items',
        description: 'Forgets what you picked',
        keywords: ['history', 'reset'],
        run: ({ recents, toast }) => {
            recents.clear();
            toast('Recent items cleared');
        },
    },
];
