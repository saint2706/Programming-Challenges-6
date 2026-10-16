import { mount } from 'svelte';
import './app.css';
import App from './App.svelte';
import { HeadSession } from './headcoupled/session.svelte.ts';
import { createTracker } from './headcoupled/tracker.ts';
import { Studio } from './studio.svelte.ts';

const params = new URLSearchParams(location.search);
const studio = new Studio();
if (params.get('mode') === 'window') studio.mode = 'window';
const head = new HeadSession();

// `?e2e=1` hands the browser tests the objects the page is made of; nothing else reads it.
if (params.has('e2e')) (window as unknown as { __app: unknown }).__app = { studio, head, createTracker };

mount(App, {
    target: document.getElementById('app')!,
    props: { studio, head, base: new URL('./', location.href).href, forceWebGL: params.get('renderer') === 'webgl' },
});
