<script lang="ts">
    import { type HeadSession } from '../headcoupled/session.svelte.ts';
    import { IRIS } from '../headcoupled/pose.ts';
    import { type Studio } from '../studio.svelte.ts';

    let { head }: { head: HeadSession; studio: Studio } = $props();
    let box = $state<HTMLElement>();
    let overlay = $state<HTMLCanvasElement>();
    const on = $derived(head.status === 'starting' || head.status === 'tracking' || head.status === 'no-face');
    const label = $derived(head.status === 'tracking' ? 'Tracking' : head.status === 'no-face' ? 'No face found' : 'Starting');

    // The video the tracker reads is the one shown, so what the person sees is what the model sees.
    $effect(() => {
        if (!box || !on) return;
        box.prepend(head.video);
        return () => head.video.remove();
    });

    // The landmarks the model found, drawn over the picture so that it is plain whether it has the face.
    $effect(() => {
        if (!overlay || !on) return;
        const canvas = overlay;
        const g = canvas.getContext('2d')!;
        let raf = 0;
        const draw = () => {
            raf = requestAnimationFrame(draw);
            const w = Math.round(canvas.clientWidth * devicePixelRatio);
            const h = Math.round(canvas.clientHeight * devicePixelRatio);
            if (canvas.width !== w || canvas.height !== h) {
                canvas.width = w;
                canvas.height = h;
            }
            g.clearRect(0, 0, w, h);
            const lm = head.landmarks;
            if (!lm) return;
            g.fillStyle = 'rgba(120, 220, 255, 0.7)';
            const dot = Math.max(1.5, w / 260);
            for (const p of lm) g.fillRect(p.x * w - dot / 2, p.y * h - dot / 2, dot, dot);
            const right = lm[IRIS.rightCenter];
            const left = lm[IRIS.leftCenter];
            if (!right || !left) return;
            g.strokeStyle = '#ffd24a';
            g.fillStyle = '#ffd24a';
            g.lineWidth = Math.max(1.5, w / 200);
            g.beginPath();
            g.moveTo(right.x * w, right.y * h);
            g.lineTo(left.x * w, left.y * h);
            g.stroke();
            for (const centre of [right, left]) {
                g.beginPath();
                g.arc(centre.x * w, centre.y * h, Math.max(3, w / 70), 0, Math.PI * 2);
                g.stroke();
            }
            g.beginPath();
            g.arc(((right.x + left.x) / 2) * w, ((right.y + left.y) / 2) * h, Math.max(2.5, w / 90), 0, Math.PI * 2);
            g.fill();
        };
        draw();
        return () => cancelAnimationFrame(raf);
    });
</script>

{#if on}
    <div class="inset" role="group" aria-label="Camera preview">
        <div class="view" bind:this={box} style:aspect-ratio={head.frameSize.width ? `${head.frameSize.width} / ${head.frameSize.height}` : '16 / 9'}>
            <canvas bind:this={overlay} aria-hidden="true"></canvas>
        </div>
        <div class="bar">
            <span class="live" aria-hidden="true"></span>
            <span class="what">Camera on · {label}</span>
            <button class="stop" onclick={() => head.stop()}>Stop</button>
        </div>
    </div>
{/if}

<style>
    .inset {
        position: absolute;
        left: 12px;
        bottom: 12px;
        width: min(220px, 42%);
        border-radius: 12px;
        overflow: hidden;
        background: #000;
        color: #fff;
        box-shadow: 0 6px 24px rgb(0 0 0 / 0.35);
    }
    .view {
        position: relative;
        background: #111;
    }
    .view canvas {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        transform: scaleX(-1);
    }
    .bar {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 6px 8px;
        font-size: 0.78rem;
    }
    .live {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #ff3b30;
        flex: none;
    }
    .what {
        flex: 1;
        min-width: 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .stop {
        border: 1px solid #fff;
        background: transparent;
        color: #fff;
        border-radius: 999px;
        padding: 2px 10px;
        min-height: 28px;
    }
</style>
