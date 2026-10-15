/*
 * Turns any JavaScript value into a short, safe string for the console. It runs in the preview's
 * sandbox next to code we do not control, so it must never throw, never run user getters, and never
 * produce more than a bounded amount of text. Loaded as a classic script (no modules in a sandbox
 * without an origin), and also loaded by the unit tests.
 */
(function (global) {
    'use strict';

    var MAX_DEPTH = 3;
    var MAX_ITEMS = 100;
    var MAX_KEYS = 50;
    var MAX_STRING = 10000;
    var MAX_NESTED_STRING = 200;
    var MAX_TOTAL = 20000;

    var IDENT = /^[A-Za-z_$][A-Za-z0-9_$]*$/;

    function str(value, limit) {
        var s = String(value);
        return s.length > limit ? s.slice(0, limit) + '… (' + (s.length - limit) + ' more characters)' : s;
    }

    function quote(s) {
        return "'" + str(s, MAX_NESTED_STRING).replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/\n/g, '\\n') + "'";
    }

    function typeName(value) {
        try {
            var proto = Object.getPrototypeOf(value);
            if (proto === null) return '[Object: null prototype]';
            var ctor = Object.getOwnPropertyDescriptor(proto, 'constructor');
            return ctor && typeof ctor.value === 'function' && ctor.value.name ? ctor.value.name : 'Object';
        } catch (e) {
            return 'Object';
        }
    }

    function describeElement(el) {
        var s = '<' + el.localName;
        var id = el.getAttribute && el.getAttribute('id');
        var cls = el.getAttribute && el.getAttribute('class');
        if (id) s += ' id="' + str(id, 40) + '"';
        if (cls) s += ' class="' + str(cls, 60) + '"';
        return s + '>';
    }

    /** `state.total` is how many characters have been produced so far across the whole value. */
    function inspect(value, depth, seen, state) {
        if (state.total > MAX_TOTAL) return '…';
        var out = inspectInner(value, depth, seen, state);
        state.total += out.length;
        return out;
    }

    function inspectInner(value, depth, seen, state) {
        try {
            switch (typeof value) {
                case 'string':
                    return depth === 0 ? str(value, MAX_STRING) : quote(value);
                case 'number':
                    return Object.is(value, -0) ? '-0' : String(value);
                case 'bigint':
                    return value + 'n';
                case 'boolean':
                case 'undefined':
                    return String(value);
                case 'symbol':
                    return value.toString();
                case 'function': {
                    var src = '';
                    try {
                        src = Function.prototype.toString.call(value);
                    } catch (e) {
                        src = '';
                    }
                    var name = value.name ? ': ' + value.name : ' (anonymous)';
                    return /^class[\s{]/.test(src) ? '[class' + (value.name ? ' ' + value.name : ' (anonymous)') + ']' : '[Function' + name + ']';
                }
            }
            if (value === null) return 'null';

            if (seen.indexOf(value) >= 0) return '[Circular]';

            var tag = Object.prototype.toString.call(value);
            if (value instanceof Error) {
                if (depth === 0) return str(value.stack || value.name + ': ' + value.message, MAX_STRING);
                return '[' + (value.name || 'Error') + ': ' + str(value.message, MAX_NESTED_STRING) + ']';
            }
            if (value instanceof Date) return isNaN(value) ? 'Invalid Date' : value.toISOString();
            if (value instanceof RegExp) return String(value);
            if (tag === '[object Window]') return '[object Window]';
            if (typeof Node !== 'undefined' && value instanceof Node) {
                return value.nodeType === 1 ? describeElement(value) : '[' + (value.nodeName || 'Node') + ']';
            }
            if (typeof Promise !== 'undefined' && value instanceof Promise) return 'Promise { <state unknown> }';
            if (value instanceof WeakMap || value instanceof WeakSet) return typeName(value) + ' { <items not enumerable> }';
            if (typeof ArrayBuffer !== 'undefined' && value instanceof ArrayBuffer) return 'ArrayBuffer { byteLength: ' + value.byteLength + ' }';

            if (depth >= MAX_DEPTH) return Array.isArray(value) ? '[Array]' : '[' + typeName(value) + ']';

            seen.push(value);
            try {
                if (Array.isArray(value) || (typeof ArrayBuffer !== 'undefined' && ArrayBuffer.isView(value) && !(value instanceof DataView))) {
                    var items = [];
                    var n = Math.min(value.length, MAX_ITEMS);
                    for (var i = 0; i < n; i++) {
                        items.push(i in value || !Array.isArray(value) ? inspect(value[i], depth + 1, seen, state) : '<empty>');
                    }
                    if (value.length > n) items.push('… ' + (value.length - n) + ' more items');
                    var head = Array.isArray(value) ? '' : typeName(value) + '(' + value.length + ') ';
                    return head + (items.length ? '[ ' + items.join(', ') + ' ]' : '[]');
                }
                if (value instanceof Map) {
                    var entries = [];
                    var count = 0;
                    value.forEach(function (v, k) {
                        if (count++ < MAX_ITEMS) entries.push(inspect(k, depth + 1, seen, state) + ' => ' + inspect(v, depth + 1, seen, state));
                    });
                    if (value.size > MAX_ITEMS) entries.push('… ' + (value.size - MAX_ITEMS) + ' more items');
                    return 'Map(' + value.size + ') ' + (entries.length ? '{ ' + entries.join(', ') + ' }' : '{}');
                }
                if (value instanceof Set) {
                    var members = [];
                    var seenCount = 0;
                    value.forEach(function (v) {
                        if (seenCount++ < MAX_ITEMS) members.push(inspect(v, depth + 1, seen, state));
                    });
                    if (value.size > MAX_ITEMS) members.push('… ' + (value.size - MAX_ITEMS) + ' more items');
                    return 'Set(' + value.size + ') ' + (members.length ? '{ ' + members.join(', ') + ' }' : '{}');
                }

                var keys = Reflect.ownKeys(value);
                var parts = [];
                for (var k = 0; k < keys.length && k < MAX_KEYS; k++) {
                    var key = keys[k];
                    var desc = Object.getOwnPropertyDescriptor(value, key);
                    if (!desc || (desc.enumerable === false && typeof key === 'string')) continue;
                    var label = typeof key === 'symbol' ? '[' + key.toString() + ']' : IDENT.test(key) ? key : quote(key);
                    // A getter is shown as a getter: running it could do anything, including hang.
                    var shown = 'get' in desc && desc.get ? (desc.set ? '[Getter/Setter]' : '[Getter]') : 'set' in desc && desc.set ? '[Setter]' : inspect(desc.value, depth + 1, seen, state);
                    parts.push(label + ': ' + shown);
                }
                if (keys.length > MAX_KEYS) parts.push('… ' + (keys.length - MAX_KEYS) + ' more properties');
                var name = typeName(value);
                var prefix = name === 'Object' ? '' : name + ' ';
                return prefix + (parts.length ? '{ ' + parts.join(', ') + ' }' : '{}');
            } finally {
                seen.pop();
            }
        } catch (e) {
            return '[Uninspectable]';
        }
    }

    function format(value) {
        return inspect(value, 0, [], { total: 0 });
    }

    /**
     * What console.log(...args) would print, including the printf-style first argument:
     * %s %d %i %f %o %O %j and %c (ignored), and %% for a percent sign.
     */
    function formatArgs(args) {
        var list = Array.prototype.slice.call(args);
        var first = list[0];
        var parts = [];
        if (typeof first === 'string' && first.indexOf('%') >= 0 && list.length > 1) {
            var used = 1;
            var text = first.replace(/%([sdifoOjc%])/g, function (m, kind) {
                if (kind === '%') return '%';
                if (used >= list.length) return m;
                var arg = list[used++];
                switch (kind) {
                    case 's':
                        return typeof arg === 'string' ? arg : format(arg);
                    case 'd':
                    case 'i':
                        return typeof arg === 'bigint' ? arg + 'n' : String(kind === 'i' ? parseInt(arg, 10) : Number(arg));
                    case 'f':
                        return String(parseFloat(arg));
                    case 'j':
                        try {
                            return JSON.stringify(arg);
                        } catch (e) {
                            return '[Circular]';
                        }
                    case 'c':
                        return '';
                    default:
                        return format(arg);
                }
            });
            parts.push(text);
            list = list.slice(used);
        } else if (list.length) {
            parts.push(format(list.shift()));
        }
        for (var i = 0; i < list.length; i++) parts.push(format(list[i]));
        return str(parts.join(' '), MAX_TOTAL);
    }

    global.__pgFormat = { format: format, formatArgs: formatArgs, limits: { MAX_STRING: MAX_STRING, MAX_TOTAL: MAX_TOTAL, MAX_ITEMS: MAX_ITEMS, MAX_DEPTH: MAX_DEPTH } };
})(typeof globalThis !== 'undefined' ? globalThis : window);
