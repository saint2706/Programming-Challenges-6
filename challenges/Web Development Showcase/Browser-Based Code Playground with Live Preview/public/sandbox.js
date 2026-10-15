/*
 * The page that runs inside the preview iframe. The iframe is sandboxed without an origin, so this
 * code and the person's code can reach nothing of the playground except by postMessage.
 *
 * Protocol (parent -> here):  { t: 'run', run, html, css, code, scroll }   replace the page and run it
 *                             { t: 'css', css }                             swap the stylesheet in place
 * Protocol (here -> parent):  { t: 'ready' }, { t: 'log', ... }, { t: 'error', ... }, { t: 'csp', ... },
 *                             { t: 'scroll', x, y }, { t: 'clear', run }
 *
 * Classic script on purpose: a sandbox without an origin cannot load module scripts from a server.
 */
(function () {
    'use strict';

    // This page is a normal file on the playground's own origin. Another site could embed it in an
    // iframe WITHOUT the sandbox attribute and post it code to run with the playground's storage, so
    // it refuses to do anything unless it really is sandboxed: a sandboxed page has the opaque
    // origin, whose name is the string "null".
    if (window.origin !== 'null') {
        document.addEventListener('DOMContentLoaded', function () {
            document.body.textContent = 'This page only runs inside the playground\u2019s preview.';
        });
        return;
    }

    var parentWindow = window.parent;
    var format = globalThis.__pgFormat;

    /** The run in progress. Everything posted carries it so the parent can drop stale messages. */
    var run = 0;
    var token = '';
    var bundleUrl = '';
    var sent = 0;
    var MAX_MESSAGES = 1000;
    var groups = 0;
    var counters = Object.create(null);
    var timers = Object.create(null);
    var scrollTimer = 0;

    function post(message) {
        try {
            // The parent's origin is not knowable from a sandbox with no origin; the parent checks `event.source`.
            parentWindow.postMessage(message, '*');
        } catch (e) {
            // nothing sensible to do: the parent is gone
        }
    }

    /** Show the person's own file name for the run's bundle, and hide this script's frames. */
    function cleanStack(stack) {
        if (typeof stack !== 'string') return '';
        var text = bundleUrl ? stack.split(bundleUrl).join('bundle.js') : stack;
        return text
            .split('\n')
            .filter(function (line) {
                return line.indexOf('sandbox.js') < 0 && line.indexOf('sandbox-format.js') < 0;
            })
            .join('\n')
            .slice(0, 8000);
    }

    function log(level, text, withStack) {
        if (sent >= MAX_MESSAGES) {
            if (sent === MAX_MESSAGES) {
                sent++;
                post({ t: 'log', run: run, level: 'system', group: 0, text: 'Console output limit reached (' + MAX_MESSAGES + ' messages). The rest of this run is not shown.' });
            }
            return;
        }
        sent++;
        var message = { t: 'log', run: run, level: level, group: groups, text: text };
        if (withStack) message.stack = cleanStack(new Error().stack);
        post(message);
    }

    var original = {};
    ['log', 'info', 'debug', 'warn', 'error', 'trace', 'assert', 'clear', 'table', 'dir', 'group', 'groupCollapsed', 'groupEnd', 'time', 'timeLog', 'timeEnd', 'count', 'countReset'].forEach(function (name) {
        original[name] = console[name];
    });

    function installConsole() {
        console.log = function () { log('log', format.formatArgs(arguments)); };
        console.info = function () { log('info', format.formatArgs(arguments)); };
        console.debug = function () { log('debug', format.formatArgs(arguments)); };
        console.warn = function () { log('warn', format.formatArgs(arguments), true); };
        console.error = function () { log('error', format.formatArgs(arguments), true); };
        console.trace = function () { log('log', 'Trace: ' + format.formatArgs(arguments), true); };
        console.assert = function (condition) {
            if (condition) return;
            var rest = Array.prototype.slice.call(arguments, 1);
            log('error', 'Assertion failed' + (rest.length ? ': ' + format.formatArgs(rest) : ''), true);
        };
        console.clear = function () {
            sent = 0;
            groups = 0;
            post({ t: 'clear', run: run });
        };
        console.table = function (data) { log('log', format.format(data)); };
        console.dir = function (value) { log('log', format.format(value)); };
        console.group = console.groupCollapsed = function () {
            if (arguments.length) log('log', format.formatArgs(arguments));
            groups = Math.min(groups + 1, 10);
        };
        console.groupEnd = function () { groups = Math.max(0, groups - 1); };
        console.time = function (label) { timers[label === undefined ? 'default' : String(label)] = performance.now(); };
        console.timeLog = console.timeEnd = function (label) {
            var key = label === undefined ? 'default' : String(label);
            if (!(key in timers)) return log('warn', 'Timer "' + key + '" does not exist');
            log('log', key + ': ' + (performance.now() - timers[key]).toFixed(2) + ' ms');
        };
        console.count = function (label) {
            var key = label === undefined ? 'default' : String(label);
            counters[key] = (counters[key] || 0) + 1;
            log('log', key + ': ' + counters[key]);
        };
        console.countReset = function (label) { counters[label === undefined ? 'default' : String(label)] = 0; };

        // Dialogs are blocked by the sandbox anyway; say so rather than failing silently.
        window.alert = function (message) { log('warn', 'alert() is not available in the preview: ' + format.formatArgs([message])); };
        window.confirm = function (message) { log('warn', 'confirm() is not available in the preview: ' + format.formatArgs([message])); return false; };
        window.prompt = function (message) { log('warn', 'prompt() is not available in the preview: ' + format.formatArgs([message])); return null; };
    }

    // ---- reporting problems -----------------------------------------------------------------
    function onError(event) {
        var error = event.error;
        post({
            t: 'error',
            run: run,
            message: String((error && error.message) || event.message || 'Script error').slice(0, 2000),
            name: error && error.name ? String(error.name).slice(0, 100) : 'Error',
            stack: cleanStack(error && error.stack),
            file: cleanStack(event.filename || ''),
            line: event.lineno | 0,
            column: event.colno | 0,
        });
    }

    function onRejection(event) {
        var reason = event.reason;
        post({
            t: 'error',
            run: run,
            message: 'Unhandled promise rejection: ' + format.format(reason).slice(0, 2000),
            name: reason && reason.name ? String(reason.name).slice(0, 100) : 'Error',
            stack: cleanStack(reason && reason.stack),
            file: '',
            line: 0,
            column: 0,
        });
    }

    function onViolation(event) {
        post({
            t: 'csp',
            run: run,
            directive: String(event.effectiveDirective || event.violatedDirective || '').slice(0, 100),
            blocked: String(event.blockedURI || '').slice(0, 300),
            sample: String(event.sample || '').slice(0, 200),
        });
    }

    function onScroll() {
        if (scrollTimer) return;
        scrollTimer = setTimeout(function () {
            scrollTimer = 0;
            post({ t: 'scroll', run: run, x: Math.round(window.scrollX), y: Math.round(window.scrollY) });
        }, 150);
    }

    function onMessage(event) {
        if (event.source !== parentWindow) return; // only the playground may talk to us
        var m = event.data;
        if (!m || typeof m !== 'object') return;
        if (m.t === 'run') start(m);
        else if (m.t === 'ping' && token) post({ t: 'pong', n: m.n | 0, token: token });
        else if (m.t === 'css' && typeof m.css === 'string') swapCss(m.css);
    }

    // A page may navigate itself and nothing here can stop it: no sandbox flag or policy covers it, and
    // the Navigation API (which could cancel it) is switched off for documents with an opaque origin.
    // The playground notices the frame load again, asks it to answer a ping with the token only this
    // script holds, and removes the frame when it cannot (see preview.ts).

    // document.open() throws away the window's event listeners, so they are put back after each run.
    function listen() {
        window.removeEventListener('error', onError);
        window.removeEventListener('unhandledrejection', onRejection);
        window.removeEventListener('securitypolicyviolation', onViolation);
        window.removeEventListener('scroll', onScroll);
        window.removeEventListener('message', onMessage);
        window.addEventListener('error', onError);
        window.addEventListener('unhandledrejection', onRejection);
        window.addEventListener('securitypolicyviolation', onViolation);
        window.addEventListener('scroll', onScroll, { passive: true });
        window.addEventListener('message', onMessage);
    }

    function swapCss(css) {
        var style = document.getElementById('pg-css');
        if (!style) {
            style = document.createElement('style');
            style.id = 'pg-css';
            document.head.appendChild(style);
        }
        style.textContent = css;
    }

    function start(m) {
        if (typeof m.html !== 'string' || typeof m.css !== 'string' || typeof m.code !== 'string') return;
        if (m.html.length > 3e6 || m.css.length > 3e6 || m.code.length > 3e6) return;

        run = m.run | 0;
        token = typeof m.token === 'string' ? m.token.slice(0, 100) : '';
        sent = 0;
        groups = 0;
        counters = Object.create(null);
        timers = Object.create(null);

        if (bundleUrl) URL.revokeObjectURL(bundleUrl);
        bundleUrl = m.code ? URL.createObjectURL(new Blob([m.code], { type: 'text/javascript' })) : '';

        // Build the page as a tree, never by gluing strings, so nothing the person wrote can break out of its place.
        var csp = document.querySelector('meta[http-equiv="Content-Security-Policy"]');
        var parsed = new DOMParser().parseFromString(m.html, 'text/html');
        if (csp) {
            var meta = parsed.createElement('meta');
            meta.setAttribute('http-equiv', 'Content-Security-Policy');
            meta.setAttribute('content', csp.getAttribute('content') || '');
            parsed.head.insertBefore(meta, parsed.head.firstChild);
        }
        var style = parsed.createElement('style');
        style.id = 'pg-css';
        style.textContent = m.css;
        parsed.head.appendChild(style);
        if (bundleUrl) {
            var script = parsed.createElement('script');
            script.type = 'module';
            script.src = bundleUrl;
            parsed.body.appendChild(script);
        }

        document.open();
        document.write('<!doctype html>' + parsed.documentElement.outerHTML);
        document.close();
        listen();

        var scroll = m.scroll;
        if (scroll && typeof scroll.x === 'number' && typeof scroll.y === 'number') {
            var restore = function () { window.scrollTo(scroll.x, scroll.y); };
            restore();
            window.addEventListener('load', restore, { once: true });
        }
    }

    installConsole();
    listen();
    post({ t: 'ready' });
})();
