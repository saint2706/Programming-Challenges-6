import './style.css';
import { App } from './app.ts';

// A page that frames this one could put a decoy over "Run it" on a shared link. A <meta> policy cannot say
// `frame-ancestors`, so the page checks for itself and refuses to start when it is not the top page.
if (window.top !== window.self) {
    document.body.replaceChildren(Object.assign(document.createElement('p'), { textContent: 'The playground cannot be shown inside another page. Open it in its own tab.' }));
    throw new Error('framed');
}

const app = new App();
// A hook for the browser tests: the live objects, not a way around the sandbox.
if (new URLSearchParams(location.search).has('e2e')) Object.assign(window, { __pg: app });
void app.start();
