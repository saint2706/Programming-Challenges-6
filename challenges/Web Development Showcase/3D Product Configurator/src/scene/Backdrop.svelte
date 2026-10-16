<script lang="ts">
    import { T, useThrelte } from '@threlte/core/webgpu';
    import { NoColorSpace, type Texture, TextureLoader } from 'three';
    import { SHADOW } from '../shoe/dimensions.ts';

    let { url, strength = 0.55, position = [0, 0, 0], rotationY = 0, scale = 1 }: { url: string; strength?: number; position?: [number, number, number]; rotationY?: number; scale?: number } = $props();
    const { invalidate } = useThrelte();
    let alpha = $state.raw<Texture | null>(null);

    // The shoe's shadow on the floor is a picture drawn at build time (prepare-shoe.ts): no shadow map to tune.
    $effect(() => {
        let gone = false;
        let loaded: Texture | undefined;
        new TextureLoader().loadAsync(url).then((texture) => {
            if (gone) return texture.dispose();
            texture.colorSpace = NoColorSpace;
            loaded = texture;
            alpha = texture;
            invalidate();
        });
        return () => {
            gone = true;
            alpha = null;
            loaded?.dispose();
        };
    });
</script>

{#if alpha}
    <T.Group {position} rotation.y={rotationY} {scale}>
        <T.Mesh rotation.x={-Math.PI / 2} position.y={0.0004} renderOrder={1}>
            <T.PlaneGeometry args={[SHADOW.width, SHADOW.depth]} />
            <T.MeshBasicMaterial color="#000000" transparent opacity={strength} alphaMap={alpha} depthWrite={false} />
        </T.Mesh>
    </T.Group>
{/if}
