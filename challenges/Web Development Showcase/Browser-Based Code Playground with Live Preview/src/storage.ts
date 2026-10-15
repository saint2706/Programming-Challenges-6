import { type Project, parseProject } from './project.ts';

const KEY = 'playground:v1';

/** Storage can be blocked, full or corrupt. Nothing here may throw: the playground has to work without it. */
function area(): Storage | null {
    try {
        return window.localStorage;
    } catch {
        return null;
    }
}

const BOOTING = 'playground:booting';

/**
 * A project can hang the tab (a pattern the loop guard cannot catch, such as a catastrophic regular
 * expression). If that project is saved, it would hang every visit. So before the first run of a start
 * a marker is written, and cleared a few seconds later; a tab that froze never gets to clear it, and
 * the next start sees it and does not run the project by itself.
 */
export function markBooting(): void {
    try {
        area()?.setItem(BOOTING, String(Date.now()));
    } catch {
        // nothing to do
    }
}

export function clearBooting(): void {
    try {
        area()?.removeItem(BOOTING);
    } catch {
        // nothing to do
    }
}

export function didNotFinishBooting(): boolean {
    try {
        return area()?.getItem(BOOTING) != null;
    } catch {
        return false;
    }
}

export interface Saved {
    project: Project;
    active: string;
}

export function loadSaved(): Saved | null {
    try {
        const raw = area()?.getItem(KEY);
        if (!raw) return null;
        const data = JSON.parse(raw) as { project?: unknown; active?: unknown };
        const project = parseProject(data.project);
        if (!project) return null;
        const active = typeof data.active === 'string' && project.files.some((f) => f.name === data.active) ? data.active : project.files[0]!.name;
        return { project, active };
    } catch {
        return null;
    }
}

/** Whether it was saved. A full or blocked store is reported, not hidden. */
export function save(project: Project, active: string): boolean {
    try {
        const storage = area();
        if (!storage) return false;
        storage.setItem(KEY, JSON.stringify({ project, active }));
        return true;
    } catch {
        return false;
    }
}
