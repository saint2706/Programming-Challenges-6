<script lang="ts">
    import { T, useThrelte } from '@threlte/core/webgpu';
    import { PART_IDS } from '../parts.ts';
    import { type ShoeParts } from '../shoe/load.ts';
    import { styleMaterial } from '../shoe/style.ts';
    import { type Studio } from '../studio.svelte.ts';

    let { parts, studio, position = [0, 0, 0], rotationY = 0, scale = 1 }: { parts: ShoeParts; studio: Studio; position?: [number, number, number]; rotationY?: number; scale?: number } = $props();
    const { invalidate } = useThrelte();

    $effect(() => {
        for (const id of PART_IDS) styleMaterial(parts[id].material, studio.design.colors[id], studio.design.finishes[id]);
        invalidate();
    });
</script>

<T.Group {position} rotation.y={rotationY} {scale}>
    {#each PART_IDS as id (id)}
        <T is={parts[id]} />
    {/each}
</T.Group>
