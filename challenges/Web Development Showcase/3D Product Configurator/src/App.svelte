<script lang="ts">
    import { Canvas } from '@threlte/core/webgpu';
    import { NeutralToneMapping } from 'three';
    import { WebGPURenderer } from 'three/webgpu';
    import { type HeadSession } from './headcoupled/session.svelte.ts';
    import Scene from './scene/Scene.svelte';
    import { type Studio } from './studio.svelte.ts';
    import CameraInset from './ui/CameraInset.svelte';
    import DesignPanel from './ui/DesignPanel.svelte';
    import Hud from './ui/Hud.svelte';
    import WindowPanel from './ui/WindowPanel.svelte';

    let { studio, head, base, forceWebGL = false }: { studio: Studio; head: HeadSession; base: string; forceWebGL?: boolean } = $props();
    let stage = $state<HTMLElement>();
    let fullscreen = $state(false);

    $effect(() => {
        const onChange = () => (fullscreen = document.fullscreenElement === stage);
        document.addEventListener('fullscreenchange', onChange);
        return () => document.removeEventListener('fullscreenchange', onChange);
    });

    // The camera stops when the person leaves the window preview.
    $effect(() => {
        if (studio.mode !== 'window') head.stop();
    });

    async function toggleFullscreen() {
        if (document.fullscreenElement) await document.exitFullscreen();
        else await stage?.requestFullscreen?.().catch(() => {});
    }

    function onKey(event: KeyboardEvent) {
        if (event.target !== stage) return;
        const keys: Record<string, [number, number, number]> = { ArrowLeft: [-0.15, 0, 1], ArrowRight: [0.15, 0, 1], ArrowUp: [0, -0.1, 1], ArrowDown: [0, 0.1, 1], '+': [0, 0, 0.9], '=': [0, 0, 0.9], '-': [0, 0, 1.1], _: [0, 0, 1.1] };
        const k = keys[event.key];
        if (studio.mode === 'design' && k) {
            event.preventDefault();
            studio.orbit(...k);
        }
        if (studio.mode === 'window' && event.key.toLowerCase() === 'f') void toggleFullscreen();
    }
</script>

<div class="app" data-mode={studio.mode}>
    <header>
        <h1>Sneaker Configurator</h1>
        <div class="modes" role="radiogroup" aria-label="View">
            <button class="btn" role="radio" aria-checked={studio.mode === 'design'} onclick={() => (studio.mode = 'design')}>Design</button>
            <button class="btn" role="radio" aria-checked={studio.mode === 'window'} onclick={() => (studio.mode = 'window')}>Window preview</button>
        </div>
    </header>

    <main>
        <!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
        <section class="stage" class:full={fullscreen} bind:this={stage} tabindex="0" role="application" aria-label="3D view of the shoe" aria-describedby="stage-help" onkeydown={onKey}>
            <Canvas
                dpr={Math.min(window.devicePixelRatio || 1, 2)}
                toneMapping={NeutralToneMapping}
                renderMode="on-demand"
                shadows={false}
                createRenderer={(canvas) => new WebGPURenderer({ canvas, antialias: true, alpha: true, forceWebGL })}
            >
                <Scene {studio} {head} {base} />
            </Canvas>

            {#if !studio.ready && !studio.error}
                <p class="overlay" role="status">Loading the shoe…</p>
            {/if}
            {#if studio.error}
                <p class="overlay error" role="alert">{studio.error}</p>
            {/if}
            <p id="stage-help" class="sr-only">
                {studio.mode === 'design' ? 'Arrow keys turn the shoe, plus and minus zoom.' : 'Move the pointer or your head to look around the room. Press F for full screen.'}
            </p>

            {#if studio.mode === 'window'}
                <CameraInset {head} {studio} />
            {/if}
            {#if studio.hud}
                <Hud {studio} {head} />
            {/if}
        </section>

        <aside class="panel" aria-label={studio.mode === 'design' ? 'Design' : 'Window preview settings'}>
            {#if studio.mode === 'design'}
                <DesignPanel {studio} />
            {:else}
                <WindowPanel {studio} {head} {fullscreen} onfullscreen={toggleFullscreen} />
            {/if}
        </aside>
    </main>
</div>

<style>
    .app {
        display: grid;
        grid-template-rows: auto 1fr;
        height: 100dvh;
    }
    header {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 8px var(--gap);
        padding: 10px var(--gap);
        border-bottom: 1px solid var(--line);
        background: var(--panel);
    }
    h1 {
        margin: 0;
        font-size: 1.1rem;
        letter-spacing: 0.01em;
    }
    .modes {
        display: flex;
        gap: 8px;
    }
    main {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 360px;
        min-height: 0;
    }
    .stage {
        position: relative;
        min-height: 0;
        background: radial-gradient(ellipse at 50% 38%, var(--stage-a), var(--stage-b));
        touch-action: none;
    }
    .stage:focus-visible {
        outline-offset: -4px;
    }
    .panel {
        overflow: auto;
        padding: var(--gap);
        background: var(--panel);
        border-left: 1px solid var(--line);
    }
    .overlay {
        position: absolute;
        inset: auto 0 50% 0;
        margin: 0;
        text-align: center;
        color: var(--muted);
        pointer-events: none;
    }
    .overlay.error {
        color: var(--danger);
        padding: 0 var(--gap);
    }
    @media (max-width: 800px) {
        main {
            grid-template-columns: minmax(0, 1fr);
            grid-template-rows: minmax(280px, 52dvh) auto;
            overflow: auto;
        }
        .panel {
            border-left: 0;
            border-top: 1px solid var(--line);
            overflow: visible;
        }
        .app {
            height: auto;
            min-height: 100dvh;
        }
    }
</style>
