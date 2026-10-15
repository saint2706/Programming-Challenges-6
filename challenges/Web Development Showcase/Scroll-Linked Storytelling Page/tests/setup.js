// jsdom has no matchMedia, and GSAP calls it while it registers ScrollTrigger. "no-preference"
// matches, which is the motion-allowed branch the engine pins under.
if (typeof window !== 'undefined' && !window.matchMedia) {
    window.matchMedia = (query) => ({
        matches: query.includes('no-preference'),
        media: query,
        addEventListener() {},
        removeEventListener() {},
        addListener() {},
        removeListener() {},
    });
}
