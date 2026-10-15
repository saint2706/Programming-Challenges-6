import type { Project } from './project.ts';

export interface Template {
    id: string;
    title: string;
    summary: string;
    project: Project;
}

const page = (title: string, body: string) => `<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>${title}</title>
</head>
<body>
${body}
</body>
</html>
`;

export const TEMPLATES: readonly Template[] = [
    {
        id: 'hello',
        title: 'Hello, JavaScript',
        summary: 'HTML, CSS and a script that counts clicks. Edit the CSS and the page updates without reloading.',
        project: {
            network: false,
            files: [
                {
                    name: 'index.html',
                    code: page(
                        'Counter',
                        `    <main>
        <h1>Hello</h1>
        <p>You clicked <strong id="count">0</strong> times.</p>
        <button id="button" type="button">Click me</button>
    </main>`,
                    ),
                },
                {
                    name: 'style.css',
                    code: `body {
    font-family: system-ui, sans-serif;
    display: grid;
    place-items: center;
    min-height: 100vh;
    margin: 0;
    background: #f4f1ea;
    color: #1f2328;
}

button {
    font: inherit;
    padding: 0.6rem 1.2rem;
    border: 0;
    border-radius: 999px;
    background: #0969da;
    color: white;
    cursor: pointer;
}
`,
                },
                {
                    name: 'main.js',
                    code: `const count = document.querySelector('#count');
const button = document.querySelector('#button');

let clicks = 0;
button.addEventListener('click', () => {
    clicks += 1;
    count.textContent = String(clicks);
    console.log('clicked', clicks);
});

console.log('ready');
`,
                },
            ],
        },
    },
    {
        id: 'preact',
        title: 'Preact + TSX',
        summary: 'A component with state, written in TSX and compiled in your browser. Preact and signals are built in.',
        project: {
            network: false,
            files: [
                { name: 'index.html', code: page('Preact counter', '    <div id="app"></div>') },
                {
                    name: 'style.css',
                    code: `body {
    font-family: system-ui, sans-serif;
    margin: 2rem;
    color: #1f2328;
}

.card {
    display: inline-grid;
    gap: 0.75rem;
    padding: 1.25rem 1.5rem;
    border: 1px solid #d0d7de;
    border-radius: 12px;
}

button {
    font: inherit;
    padding: 0.4rem 0.9rem;
    border-radius: 8px;
    border: 1px solid #d0d7de;
    background: #f6f8fa;
    cursor: pointer;
}
`,
                },
                {
                    name: 'main.tsx',
                    code: `import { render } from 'preact';
import { useState } from 'preact/hooks';
import { signal } from '@preact/signals';
import { Greeting } from './greeting.tsx';

const clicks = signal(0);

function Counter({ step }: { step: number }) {
    const [local, setLocal] = useState(0);
    return (
        <div class="card">
            <Greeting name="Preact" />
            <p>State: {local}. Signal: {clicks}.</p>
            <button onClick={() => { setLocal(local + step); clicks.value += step; }}>
                Add {step}
            </button>
        </div>
    );
}

render(<Counter step={1} />, document.getElementById('app')!);
`,
                },
                {
                    name: 'greeting.tsx',
                    code: `export function Greeting({ name }: { name: string }) {
    return <h1>Hello from {name}</h1>;
}
`,
                },
            ],
        },
    },
    {
        id: 'typescript',
        title: 'TypeScript',
        summary: 'Types are checked by you and erased by esbuild. A tiny todo list with an exhaustive switch.',
        project: {
            network: false,
            files: [
                { name: 'index.html', code: page('Todos', '    <ul id="list"></ul>\n    <form id="form"><input id="text" placeholder="Add a task" autocomplete="off"> <button>Add</button></form>') },
                { name: 'style.css', code: 'body { font-family: system-ui, sans-serif; margin: 2rem; }\nli.done { text-decoration: line-through; color: #6e7781; }\n' },
                {
                    name: 'main.ts',
                    code: `import { add, toggle, summary, type Todo } from './todos.ts';

let todos: Todo[] = [];
const list = document.querySelector<HTMLUListElement>('#list')!;
const form = document.querySelector<HTMLFormElement>('#form')!;
const text = document.querySelector<HTMLInputElement>('#text')!;

function draw(): void {
    list.replaceChildren(
        ...todos.map((t) => {
            const li = document.createElement('li');
            li.textContent = t.title;
            li.classList.toggle('done', t.done);
            li.onclick = () => { todos = toggle(todos, t.id); draw(); };
            return li;
        }),
    );
    console.log(summary(todos));
}

form.addEventListener('submit', (e) => {
    e.preventDefault();
    if (text.value.trim()) todos = add(todos, text.value.trim());
    text.value = '';
    draw();
});

draw();
`,
                },
                {
                    name: 'todos.ts',
                    code: `export interface Todo {
    id: number;
    title: string;
    done: boolean;
}

let next = 1;

export const add = (todos: Todo[], title: string): Todo[] => [...todos, { id: next++, title, done: false }];

export const toggle = (todos: Todo[], id: number): Todo[] => todos.map((t) => (t.id === id ? { ...t, done: !t.done } : t));

export function summary(todos: Todo[]): string {
    const done = todos.filter((t) => t.done).length;
    return \`\${done} of \${todos.length} done\`;
}
`,
                },
            ],
        },
    },
    {
        id: 'canvas',
        title: 'Canvas animation',
        summary: 'A requestAnimationFrame loop. Edit the numbers and watch the page reload with its scroll position kept.',
        project: {
            network: false,
            files: [
                { name: 'index.html', code: page('Canvas', '    <canvas id="c" width="640" height="360"></canvas>') },
                { name: 'style.css', code: 'body { margin: 0; background: #0d1117; display: grid; place-items: center; min-height: 100vh; }\ncanvas { max-width: 100%; border-radius: 12px; }\n' },
                {
                    name: 'main.js',
                    code: `const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const dots = Array.from({ length: 60 }, (_, i) => ({ a: i * 0.35, r: 20 + i * 4, hue: i * 6 }));

function frame(t) {
    ctx.fillStyle = 'rgba(13, 17, 23, 0.25)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    for (const d of dots) {
        const x = canvas.width / 2 + Math.cos(d.a + t / 1500) * d.r * 1.6;
        const y = canvas.height / 2 + Math.sin(d.a + t / 1500) * d.r * 0.8;
        ctx.fillStyle = \`hsl(\${d.hue + t / 20}, 80%, 60%)\`;
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
    }
    requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
`,
                },
            ],
        },
    },
];

export const DEFAULT_TEMPLATE = TEMPLATES[0]!;
