/**
 * What the preview may say to the playground. The preview runs code we did not write, so anything
 * it posts is untrusted: this turns a posted value into one of a few known shapes with bounded
 * sizes, or into `null`. The playground never reads a raw message anywhere else.
 */

export type Level = 'log' | 'info' | 'debug' | 'warn' | 'error' | 'system';

export type SandboxEvent =
    | { t: 'ready' }
    | { t: 'pong'; n: number; token: string }
    | { t: 'clear'; run: number }
    | { t: 'log'; run: number; level: Level; group: number; text: string; stack: string }
    | { t: 'error'; run: number; name: string; message: string; stack: string; file: string; line: number; column: number }
    | { t: 'csp'; run: number; directive: string; blocked: string; sample: string }
    | { t: 'scroll'; run: number; x: number; y: number };

const LEVELS: ReadonlySet<string> = new Set(['log', 'info', 'debug', 'warn', 'error', 'system']);

const text = (value: unknown, max: number): string => (typeof value === 'string' ? value.slice(0, max) : '');
const int = (value: unknown, min: number, max: number): number =>
    typeof value === 'number' && Number.isFinite(value) ? Math.min(max, Math.max(min, Math.trunc(value))) : min;

export function parseSandboxEvent(data: unknown): SandboxEvent | null {
    if (typeof data !== 'object' || data === null) return null;
    const m = data as Record<string, unknown>;
    const run = int(m.run, 0, 2 ** 31 - 1);
    switch (m.t) {
        case 'ready':
            return { t: 'ready' };
        case 'pong':
            return { t: 'pong', n: int(m.n, 0, 2 ** 31 - 1), token: text(m.token, 100) };
        case 'clear':
            return { t: 'clear', run };
        case 'log': {
            if (typeof m.level !== 'string' || !LEVELS.has(m.level)) return null;
            return { t: 'log', run, level: m.level as Level, group: int(m.group, 0, 10), text: text(m.text, 20_000), stack: text(m.stack, 8000) };
        }
        case 'error':
            return {
                t: 'error',
                run,
                name: text(m.name, 100) || 'Error',
                message: text(m.message, 2000),
                stack: text(m.stack, 8000),
                file: text(m.file, 300),
                line: int(m.line, 0, 10_000_000),
                column: int(m.column, 0, 10_000_000),
            };
        case 'csp':
            return { t: 'csp', run, directive: text(m.directive, 100), blocked: text(m.blocked, 300), sample: text(m.sample, 200) };
        case 'scroll':
            return { t: 'scroll', run, x: int(m.x, 0, 10_000_000), y: int(m.y, 0, 10_000_000) };
        default:
            return null;
    }
}
