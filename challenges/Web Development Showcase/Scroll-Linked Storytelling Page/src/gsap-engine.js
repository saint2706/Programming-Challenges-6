// The GSAP engine: ScrollTrigger pins the chart and stripe stages and scrubs their animations.
// Loaded with a dynamic import, so browsers that stay on the CSS engine never download GSAP.
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const MOTION = '(prefers-reduced-motion: no-preference)';

/**
 * Starts every GSAP animation. Returns a handle whose revert() removes all of them (triggers,
 * pins, inline styles); the story calls it before switching back to CSS.
 * With reduced motion nothing is created, and the CSS layout is static as well.
 */
export function mountGsapEngine({ scrolly, stage, path, progressBar, stripesTrack, stripesStage, strip }) {
    const mm = gsap.matchMedia();
    mm.add({ motion: MOTION }, () => {
        // pinSpacing: false because the track elements already have the scroll height (CSS sets it).
        gsap.set(path, { strokeDasharray: 1, strokeDashoffset: 1 });
        gsap.to(path, {
            strokeDashoffset: 0,
            ease: 'none',
            scrollTrigger: {
                trigger: scrolly,
                start: 'top top',
                end: 'bottom bottom',
                scrub: 1,
                pin: stage,
                pinSpacing: false,
            },
        });

        gsap.to(progressBar, {
            scaleX: 1,
            ease: 'none',
            scrollTrigger: { trigger: document.documentElement, start: 0, end: 'max', scrub: true },
        });

        gsap.fromTo(
            strip,
            { x: 0 },
            {
                x: () => window.innerWidth - strip.offsetWidth,
                ease: 'none',
                scrollTrigger: {
                    trigger: stripesTrack,
                    start: 'top top',
                    end: 'bottom bottom',
                    scrub: 1,
                    pin: stripesStage,
                    pinSpacing: false,
                    invalidateOnRefresh: true,
                },
            },
        );
    });

    return {
        triggerCount: () => ScrollTrigger.getAll().length,
        revert: () => mm.revert(),
    };
}
