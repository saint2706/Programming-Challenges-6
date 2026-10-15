import * as esbuild from 'esbuild';
import vendor from 'virtual:vendor';
import { beforeEach, describe, expect, test } from 'vitest';
import { bundleProject } from '../src/bundle.ts';
import { GUARD_RUNTIME, LOOP_LIMIT_MS, createGuard } from '../src/guardruntime.ts';

describe('the loop guard (what every guarded loop calls)', () => {
    let clock = 0;
    let pending: (() => void)[] = [];
    let guard: () => void;

    beforeEach(() => {
        clock = 1000;
        pending = [];
        guard = createGuard(LOOP_LIMIT_MS, () => clock, (fn) => void pending.push(fn));
    });

    /** The event loop turns: a new synchronous run begins. */
    const turn = () => pending.splice(0).forEach((fn) => fn());
    /** One synchronous run: `n` iterations while the clock advances `step` ms per iteration. */
    const spin = (n: number, step: number) => {
        for (let i = 0; i < n; i++) {
            clock += step;
            guard();
        }
    };

    test('short loops, even very long ones that finish quickly, are left alone', () => {
        expect(() => spin(1_000_000, 0)).not.toThrow();
        turn();
        expect(() => spin(100, 10)).not.toThrow(); // a second of slow iterations
    });

    test('a loop that goes on for more than two seconds is stopped, with a message that explains', () => {
        expect(() => spin(100_000, 1)).toThrow(/Stopped a loop that ran for over 2 seconds/);
        turn();
        expect(() => spin(100_000, 1)).toThrow(/probably infinite/);
    });

    test('it trips within a few iterations of the limit, not long after', () => {
        let iterations = 0;
        try {
            for (;;) {
                clock += 1;
                iterations++;
                guard();
            }
        } catch {
            // expected
        }
        expect(iterations).toBeGreaterThan(2000);
        expect(iterations).toBeLessThan(2000 + 40);
    });

    test('once tripped it stays tripped for the rest of that run, so catching the error and looping on still ends', () => {
        expect(() => spin(100_000, 1)).toThrow();
        expect(() => spin(100, 0)).toThrow(/ran for over/);
        expect(() => spin(100, 0)).toThrow(/ran for over/);
    });

    test('and it is a fresh start for the next run (the next macrotask)', () => {
        expect(() => spin(100_000, 1)).toThrow();
        turn();
        expect(() => spin(100, 1)).not.toThrow();
    });

    test('time spent between runs does not count against a loop', () => {
        spin(50, 20); // a run of one second
        turn();
        clock += 60_000;
        expect(() => spin(50, 20)).not.toThrow(); // another, a minute later
    });

    test('tight loops read the clock rarely, which is what keeps the guard cheap', () => {
        let reads = 0;
        const counting = createGuard(LOOP_LIMIT_MS, () => (reads++, clock), (fn) => void pending.push(fn));
        for (let i = 0; i < 1_000_000; i++) counting(); // the clock never moves: every iteration is "instant"
        expect(reads).toBeLessThan(1_000_000 / 1023 + 20);
        expect(reads).toBeGreaterThan(900); // but it does still look, regularly
    });

    test('slow iterations are looked at often, so a stuck loop with a heavy body is stopped on time', () => {
        let iterations = 0;
        const t0 = clock;
        try {
            for (;;) {
                clock += 5; // each iteration is slow
                iterations++;
                guard();
            }
        } catch {
            // expected
        }
        expect(iterations).toBeLessThan(2000 / 5 + 40);
        expect(clock - t0).toBeLessThan(LOOP_LIMIT_MS + 200);
    });

    test('a loop that was fast and turns slow is still stopped within a second of the limit', () => {
        spin(500_000, 0); // fast: the mask widens to its maximum
        turn();
        const t0 = clock;
        expect(() => spin(1_000_000, 1)).toThrow(/Stopped a loop/);
        expect(clock - t0).toBeLessThan(LOOP_LIMIT_MS + 1100);
    });

    test('it asks to be told when the run is over only once per run', () => {
        spin(1000, 0);
        expect(pending).toHaveLength(1);
        turn();
        spin(1000, 0);
        expect(pending).toHaveLength(1);
    });
});

describe('as it is put in a bundle', () => {
    test('the source text is self-contained: it runs with nothing but performance and setTimeout in reach', () => {
        const timers: (() => void)[] = [];
        let now = 0;
        const run = new Function('performance', 'setTimeout', `${GUARD_RUNTIME}; return __pgLoop;`);
        const guard = run({ now: () => now }, (fn: () => void) => void timers.push(fn)) as () => void;
        expect(() => {
            for (let i = 0; i < 100_000; i++) {
                now += 1;
                guard();
            }
        }).toThrow(/Stopped a loop/);
        expect(timers.length).toBeGreaterThan(0);
    });

    test('every bundle starts with it, and its lines do not disturb the source map', async () => {
        const r = await bundleProject({ network: false, files: [{ name: 'index.html', code: '' }, { name: 'main.js', code: 'console.log(1);\n' }] }, esbuild, vendor);
        if (!r.ok) throw new Error(JSON.stringify(r.errors));
        expect(r.code.startsWith('var __pgLoop =')).toBe(true);
        expect(r.code).toContain('console.log(1)');
    });
});
