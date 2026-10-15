// The DOM side of the story: the engine toggle, the active step (IntersectionObserver), the chart's
// highlight band, and the mounting of the animation engine. Dependencies are passed in so jsdom tests
// can supply storage, support and an observer of their own.
import { xOf } from './chart.js';
import { formatAnomaly, rangeMean } from './gistemp.js';
import { ENGINES, otherEngine, readStoredEngine, resolveEngine, writeStoredEngine } from './engine.js';
import { pickActiveStep, stepForProgress } from './steps.js';

const LABEL = { css: 'native CSS', gsap: 'GSAP ScrollTrigger' };

/**
 * @param {object} deps
 * @param {Document} deps.doc
 * @param {Storage | null} deps.storage
 * @param {boolean} deps.cssSupported
 * @param {{ year: number, value: number }[]} deps.years
 * @param {() => Promise<{ mountGsapEngine: Function }>} deps.loadGsap
 * @param {typeof IntersectionObserver} [deps.IO]
 */
export function mountStory({ doc, storage, cssSupported, years, loadGsap, IO = globalThis.IntersectionObserver }) {
    const root = doc.documentElement;
    const toggle = doc.querySelector('#engine-toggle');
    const scrolly = doc.querySelector('#chart-story');
    const steps = [...doc.querySelectorAll('[data-step]')];
    const jumps = [...doc.querySelectorAll('[data-jump]')];
    const band = doc.querySelector('.chart-svg .band');
    const note = doc.querySelector('.chart-svg .chart-note');

    let engine = resolveEngine({ stored: readStoredEngine(storage), cssSupported });
    let handle = null;
    let token = 0; // a toggle that is superseded while GSAP is still loading must not mount

    const els = () => ({
        scrolly,
        stage: doc.querySelector('#chart-stage'),
        path: doc.querySelector('.chart-line'),
        progressBar: doc.querySelector('.progress-bar'),
        stripesTrack: doc.querySelector('#stripes-story'),
        stripesStage: doc.querySelector('#stripes-stage'),
        strip: doc.querySelector('.strip'),
    });

    function updateToggle() {
        if (!toggle) return;
        toggle.hidden = false;
        if (!cssSupported) {
            toggle.textContent = `Engine: ${LABEL.gsap} (native scroll animations are not supported here)`;
            toggle.disabled = true;
            return;
        }
        toggle.disabled = false;
        toggle.textContent = `Engine: ${LABEL[engine]}. Switch to ${LABEL[otherEngine(engine)]}`;
    }

    async function apply(next) {
        const mine = ++token;
        engine = next;
        root.dataset.engine = next;
        handle?.revert();
        handle = null;
        updateToggle();
        if (next === 'gsap') {
            const mod = await loadGsap();
            if (mine !== token) return;
            handle = mod.mountGsapEngine(els());
        }
    }

    /** The user's choice: switches in place (no reload) and remembers it. */
    function setEngine(next) {
        if (!ENGINES.includes(next)) throw new RangeError(`unknown engine: ${next}`);
        if (next === 'css' && !cssSupported) return Promise.resolve(engine);
        writeStoredEngine(storage, next);
        return apply(next).then(() => engine);
    }

    toggle?.addEventListener('click', () => setEngine(otherEngine(engine)));

    // The chart highlight follows the active step: its years become a band and a note.
    function highlight(step) {
        const from = Number(step.dataset.from);
        const to = Number(step.dataset.to);
        if (band) {
            band.setAttribute('x', xOf(from).toFixed(1));
            band.setAttribute('width', (xOf(to) - xOf(from)).toFixed(1));
        }
        if (note) {
            const mean = rangeMean(years, from, to);
            note.textContent = `${from} to ${to}: average ${formatAnomaly(mean)}`;
        }
    }

    let current = null;
    function setCurrent(id) {
        if (id === current) return;
        current = id;
        for (const step of steps) {
            if (step.id === id) step.setAttribute('aria-current', 'step');
            else step.removeAttribute('aria-current');
        }
        for (const link of jumps) {
            if (link.dataset.jump === id) link.setAttribute('aria-current', 'step');
            else link.removeAttribute('aria-current');
        }
        const active = steps.find((s) => s.id === id);
        if (active) highlight(active);
    }

    if (IO) {
        const state = steps.map((s) => ({ id: s.id, isIntersecting: false }));
        const observer = new IO(
            (entries) => {
                for (const entry of entries) {
                    const rec = state.find((r) => r.id === entry.target.id);
                    if (rec) rec.isIntersecting = entry.isIntersecting;
                }
                setCurrent(pickActiveStep(state, current));
            },
            { rootMargin: '-50% 0px -50% 0px' },
        );
        for (const step of steps) observer.observe(step);
    } else if (scrolly) {
        // No IntersectionObserver: map the scroll position through the story to a step.
        const onScroll = () => {
            const rect = scrolly.getBoundingClientRect();
            const span = rect.height - root.clientHeight;
            const progress = span <= 0 ? 0 : -rect.top / span;
            const index = stepForProgress(progress, steps.length);
            if (steps[index]) setCurrent(steps[index].id);
        };
        globalThis.addEventListener('scroll', onScroll, { passive: true });
        onScroll();
    }

    updateToggle();
    if (engine === 'gsap') apply('gsap');
    else root.dataset.engine = engine;

    return {
        get engine() {
            return engine;
        },
        setEngine,
        /** ScrollTriggers created by the current GSAP mount (0 when on CSS). */
        triggerCount: () => handle?.triggerCount() ?? 0,
        /** Test and debug hook: the step that aria-current points at. */
        currentStep: () => current,
    };
}
