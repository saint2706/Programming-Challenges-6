<script lang="ts">
    import { type HeadSession } from '../headcoupled/session.svelte.ts';
    import { type Studio } from '../studio.svelte.ts';

    let { studio, head }: { studio: Studio; head: HeadSession } = $props();

    // What the picture is drawn for changes every frame and is not reactive; read it a few times a second.
    let seen = $state({ eye: { x: 0, y: 0, z: 0 }, width: 0, height: 0, source: 'pointer' as string });
    $effect(() => {
        const id = setInterval(() => (seen = { ...studio.seen, eye: { ...studio.seen.eye } }), 250);
        return () => clearInterval(id);
    });
    const cm = (m: number) => (m * 100).toFixed(1);
</script>

<div class="hud" role="region" aria-label="Numbers">
    <dl>
        <dt>Drawing</dt>
        <dd>{studio.backend || '…'} · {studio.fps} fps</dd>
        <dt>Picture follows</dt>
        <dd>{seen.source}</dd>
        <dt>Eye (x, y, z)</dt>
        <dd>{cm(seen.eye.x)}, {cm(seen.eye.y)}, {cm(seen.eye.z)} cm</dd>
        <dt>Window</dt>
        <dd>{cm(seen.width)} × {cm(seen.height)} cm · shoe {Math.round(studio.shoeScale * 100)}%</dd>
        <dt>Camera</dt>
        <dd>{head.status}{head.delegate ? ` · ${head.delegate}` : ''} · {head.fps} fps · {head.inferenceMs} ms</dd>
        <dt>Head</dt>
        <dd>{head.distanceCm} cm away · turned {head.yawDeg}°</dd>
        <dt>Pupil distance</dt>
        <dd>{head.ipdMm} mm {head.ipdLearned ? '(measured)' : '(average)'}</dd>
    </dl>
</div>

<style>
    .hud {
        position: absolute;
        top: 12px;
        left: 12px;
        max-width: calc(100% - 24px);
        padding: 8px 12px;
        border-radius: 10px;
        background: rgb(10 12 16 / 0.78);
        color: #e8ebf1;
        font: 12px/1.5 ui-monospace, Consolas, monospace;
        pointer-events: none;
    }
    dl {
        display: grid;
        grid-template-columns: auto 1fr;
        gap: 0 12px;
        margin: 0;
    }
    dt {
        color: #9aa3b2;
    }
    dd {
        margin: 0;
    }
</style>
