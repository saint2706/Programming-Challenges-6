<script lang="ts">
    import { T, useTask, useThrelte } from '@threlte/core/webgpu';
    import { type WebGPURenderer } from 'three/webgpu';
    import { type HeadSession } from '../headcoupled/session.svelte.ts';
    import { type ShoeParts, loadShoe } from '../shoe/load.ts';
    import { type Studio } from '../studio.svelte.ts';
    import Backdrop from './Backdrop.svelte';
    import DesignRig from './DesignRig.svelte';
    import Environment from './Environment.svelte';
    import ShoeModel from './ShoeModel.svelte';
    import WindowRig from './WindowRig.svelte';

    let { studio, head, base }: { studio: Studio; head: HeadSession; base: string } = $props();
    const { renderer, renderMode, shouldRender } = useThrelte();
    let parts = $state.raw<ShoeParts | null>(null);

    $effect(() => {
        studio.memory = () => {
            const { geometries, textures } = (renderer as WebGPURenderer).info.memory;
            return { geometries, textures };
        };
    });

    $effect(() => {
        studio.backend = (renderer as WebGPURenderer).backend && ((renderer as WebGPURenderer).backend as unknown as { isWebGPUBackend?: boolean }).isWebGPUBackend ? 'WebGPU' : 'WebGL 2';
    });

    // The window preview redraws every frame (the picture follows the head); the design view only when something changes.
    $effect(() => {
        renderMode.set(studio.mode === 'window' ? 'always' : 'on-demand');
    });

    $effect(() => {
        let gone = false;
        loadShoe(new URL('models/shoe.glb', base).href).then(
            (p) => {
                if (!gone) parts = p;
            },
            (error: unknown) => {
                studio.error = `The shoe could not load: ${error instanceof Error ? error.message : String(error)}`;
            },
        );
        return () => {
            gone = true;
        };
    });

    // `frames` counts frames that are drawn, so it stands still while the design view is idle.
    let sinceReport = 0;
    let drawnAtReport = 0;
    let drawn = 0;
    useTask(
        (delta) => {
            if (shouldRender()) drawn++;
            studio.frames = drawn;
            if (drawn > 2 && parts) studio.ready = true;
            sinceReport += delta;
            if (sinceReport >= 0.5) {
                studio.fps = Math.round((drawn - drawnAtReport) / sinceReport);
                sinceReport = 0;
                drawnAtReport = drawn;
            }
        },
        { autoInvalidate: false },
    );
</script>

<Environment url={new URL('env/studio_small_08_1k.hdr', base).href} intensity={studio.mode === 'window' ? 0.7 : 1} onerror={(m) => (studio.error = m)} />

{#if studio.mode === 'design'}
    <DesignRig {studio} />
    <T.DirectionalLight position={[0.4, 0.8, 0.5]} intensity={1.2} />
    <Backdrop url={new URL('models/shadow.png', base).href} />
    {#if parts}
        <ShoeModel {parts} {studio} />
    {/if}
{:else if parts}
    <WindowRig {studio} {head} {parts} {base} />
{/if}
