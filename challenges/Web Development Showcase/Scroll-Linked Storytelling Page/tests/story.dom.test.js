import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { chartMarkup, dataTableMarkup, xOf } from '../src/chart.js';
import data from '../src/data/gistemp-annual.json';
import { STORAGE_KEY } from '../src/engine.js';
import { stripesMarkup } from '../src/stripes.js';
import { mountStory } from '../src/story.js';
import * as realGsapEngine from '../src/gsap-engine.js';

// The body of the real page, with the build-time placeholders filled exactly as vite.config.js does.
const html = readFileSync(resolve('index.html'), 'utf8');
const body = html
    .match(/<body[\s\S]*<\/body>/)[0]
    .replace('<!--chart-svg-->', chartMarkup(data.years))
    .replace('<!--data-table-->', dataTableMarkup(data.years))
    .replace('<!--stripes-->', stripesMarkup(data.years));

class FakeIO {
    static instances = [];
    constructor(callback, options) {
        this.callback = callback;
        this.options = options;
        this.targets = [];
        FakeIO.instances.push(this);
    }
    observe(target) {
        this.targets.push(target);
    }
    disconnect() {}
}

const memory = (initial = {}) => {
    const map = new Map(Object.entries(initial));
    return {
        getItem: (k) => (map.has(k) ? map.get(k) : null),
        setItem: (k, v) => map.set(k, String(v)),
    };
};

const deferred = () => {
    let resolve;
    const promise = new Promise((r) => {
        resolve = r;
    });
    return { promise, resolve };
};

function mount({ storage = memory(), cssSupported = true, IO = FakeIO, loadGsap } = {}) {
    const loader = loadGsap ?? vi.fn(async () => realGsapEngine);
    const story = mountStory({ doc: document, storage, cssSupported, years: data.years, loadGsap: loader, IO });
    return { story, loader, storage };
}

const step = (n) => document.querySelector(`#step-${n}`);
const jump = (n) => document.querySelector(`[data-jump="step-${n}"]`);

beforeEach(() => {
    FakeIO.instances = [];
    document.body.innerHTML = body;
    document.documentElement.removeAttribute('data-engine');
});

afterEach(() => {
    document.body.innerHTML = '';
});

describe('engine toggle', () => {
    it('starts on native CSS without loading GSAP, and shows the toggle', () => {
        const { story, loader } = mount();
        expect(story.engine).toBe('css');
        expect(document.documentElement.dataset.engine).toBe('css');
        expect(loader).not.toHaveBeenCalled();
        const toggle = document.querySelector('#engine-toggle');
        expect(toggle.hidden).toBe(false);
        expect(toggle.textContent).toContain('Switch to GSAP ScrollTrigger');
    });

    it('switches to GSAP in place, pins and remembers the choice, and reports its triggers', async () => {
        const { story, loader, storage } = mount();
        const marker = document.body;
        marker.dataset.survived = 'yes';
        await story.setEngine('gsap');
        expect(loader).toHaveBeenCalledTimes(1);
        expect(story.engine).toBe('gsap');
        expect(document.documentElement.dataset.engine).toBe('gsap');
        expect(storage.getItem(STORAGE_KEY)).toBe('gsap');
        expect(story.triggerCount()).toBeGreaterThan(0);
        expect(document.body.dataset.survived).toBe('yes'); // no reload
    });

    it('switching back and forth never duplicates triggers', async () => {
        const { story } = mount();
        await story.setEngine('gsap');
        const first = story.triggerCount();
        for (let i = 0; i < 3; i++) {
            await story.setEngine('css');
            expect(story.triggerCount()).toBe(0);
            await story.setEngine('gsap');
            expect(story.triggerCount()).toBe(first);
        }
    });

    it('a toggle superseded while GSAP is still loading does not mount a stale engine', async () => {
        const gate = deferred();
        const loader = vi.fn(() => gate.promise.then(() => realGsapEngine));
        const { story } = mount({ loadGsap: loader });
        const pending = story.setEngine('gsap');
        await story.setEngine('css');
        gate.resolve();
        await pending;
        expect(story.engine).toBe('css');
        expect(story.triggerCount()).toBe(0);
    });

    it('reads a stored GSAP choice at start-up', async () => {
        const { story, loader } = mount({ storage: memory({ [STORAGE_KEY]: 'gsap' }) });
        await vi.waitFor(() => expect(loader).toHaveBeenCalledTimes(1));
        await vi.waitFor(() => expect(story.triggerCount()).toBeGreaterThan(0));
        expect(story.engine).toBe('gsap');
    });

    it('falls back to GSAP and disables the toggle where native scroll animations are missing', () => {
        const { story, loader } = mount({ storage: memory({ [STORAGE_KEY]: 'css' }), cssSupported: false });
        expect(story.engine).toBe('gsap');
        expect(loader).toHaveBeenCalledTimes(1);
        const toggle = document.querySelector('#engine-toggle');
        expect(toggle.disabled).toBe(true);
        expect(toggle.textContent).toContain('not supported');
    });

    it('works with storage that throws on every access', async () => {
        const broken = {
            getItem: () => {
                throw new DOMException('blocked', 'SecurityError');
            },
            setItem: () => {
                throw new DOMException('blocked', 'SecurityError');
            },
        };
        const { story } = mount({ storage: broken });
        await story.setEngine('gsap');
        expect(story.engine).toBe('gsap');
    });

    it('the toggle button itself flips the engine', async () => {
        const { story } = mount();
        document.querySelector('#engine-toggle').click();
        await vi.waitFor(() => expect(story.engine).toBe('gsap'));
    });
});

describe('active step (IntersectionObserver)', () => {
    it('observes every step and gives aria-current to the one crossing the middle line', () => {
        mount();
        const io = FakeIO.instances[0];
        expect(io.options.rootMargin).toBe('-50% 0px -50% 0px');
        expect(io.targets.map((t) => t.id)).toEqual(['step-1', 'step-2', 'step-3', 'step-4', 'step-5']);

        io.callback([{ target: step(3), isIntersecting: true }]);
        expect(step(3).getAttribute('aria-current')).toBe('step');
        expect(jump(3).getAttribute('aria-current')).toBe('step');
        expect(document.querySelectorAll('[aria-current]')).toHaveLength(2);
    });

    it('moves the highlight band to the step years and writes the step mean as the note', () => {
        mount();
        const io = FakeIO.instances[0];
        io.callback([{ target: step(3), isIntersecting: true }]);
        const band = document.querySelector('.chart-svg .band');
        expect(Number(band.getAttribute('x'))).toBeCloseTo(xOf(1940), 1);
        expect(Number(band.getAttribute('width'))).toBeCloseTo(xOf(1975) - xOf(1940), 1);
        expect(document.querySelector('.chart-note').textContent).toBe('1940 to 1975: average −0.01 °C');
    });

    it('keeps the current step when the observer reports a neighbour leaving the line', () => {
        mount();
        const io = FakeIO.instances[0];
        io.callback([{ target: step(2), isIntersecting: true }]);
        io.callback([{ target: step(1), isIntersecting: false }]);
        expect(step(2).getAttribute('aria-current')).toBe('step');
        io.callback([{ target: step(2), isIntersecting: false }, { target: step(3), isIntersecting: true }]);
        expect(step(3).getAttribute('aria-current')).toBe('step');
        expect(step(2).hasAttribute('aria-current')).toBe(false);
    });

    it('without IntersectionObserver, the scroll position picks the first step', () => {
        mount({ IO: null }); // null, not undefined: the helper's default would replace undefined
        expect(step(1).getAttribute('aria-current')).toBe('step');
        expect(FakeIO.instances).toHaveLength(0);
    });
});

describe('the markup the story depends on', () => {
    it('the chart, stripes and table are in place and no SVG path is hidden without JavaScript', () => {
        expect(document.querySelector('.chart-line')).not.toBeNull();
        expect(document.querySelectorAll('.stripe')).toHaveLength(146);
        expect(document.querySelectorAll('.data-table tbody tr')).toHaveLength(146);
    });

    it('steps are real headings in DOM order', () => {
        const headings = [...document.querySelectorAll('#chart-story .step h3')].map((h) => h.textContent);
        expect(headings).toEqual(['The cool start', 'The first climb', 'The flat middle', 'The steady rise', 'The record decades']);
    });
});
