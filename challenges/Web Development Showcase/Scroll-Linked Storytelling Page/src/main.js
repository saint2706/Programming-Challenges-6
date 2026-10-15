import data from './data/gistemp-annual.json';
import { cssTimelineSupported } from './engine.js';
import { mountStory } from './story.js';

// localStorage access itself can throw (sandboxed iframes, blocked cookies); the story copes with null.
let storage = null;
try {
    storage = globalThis.localStorage;
} catch {
    storage = null;
}

const story = mountStory({
    doc: document,
    storage,
    cssSupported: cssTimelineSupported(),
    years: data.years,
    loadGsap: () => import('./gsap-engine.js'),
});

// Read by the browser tests and the benchmark; not part of the page's behaviour.
globalThis.__storyDebug = story;
