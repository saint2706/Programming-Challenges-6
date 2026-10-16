<script lang="ts">
    import { T } from '@threlte/core/webgpu';
    import { CanvasTexture, RepeatWrapping, SRGBColorSpace } from 'three';

    let { width, height, depth = 0.9, cell = 0.05 }: { width: number; height: number; depth?: number; cell?: number } = $props();

    // One square of a grid, tiled over each wall. Lines are what make depth readable: they converge, and they slide against each other when the head moves.
    function tile(): CanvasTexture {
        const c = document.createElement('canvas');
        c.width = c.height = 128;
        const g = c.getContext('2d')!;
        g.fillStyle = '#ffffff';
        g.fillRect(0, 0, 128, 128);
        g.strokeStyle = '#8f98a6';
        g.lineWidth = 3;
        g.strokeRect(0, 0, 128, 128);
        const t = new CanvasTexture(c);
        t.colorSpace = SRGBColorSpace;
        t.wrapS = t.wrapT = RepeatWrapping;
        t.anisotropy = 8;
        return t;
    }
    const base = tile();
    function walls(w: number, h: number, d: number) {
        const repeat = (a: number, b: number) => {
            const t = base.clone();
            t.needsUpdate = true;
            t.repeat.set(a / cell, b / cell);
            return t;
        };
        return {
            back: repeat(w, h),
            floor: repeat(w, d),
            ceiling: repeat(w, d),
            side: repeat(d, h),
        };
    }
    const maps = $derived(walls(width, height, depth));

    // Every size the window takes makes new textures; the old ones go.
    $effect(() => {
        const current = maps;
        return () => {
            for (const t of Object.values(current)) t.dispose();
        };
    });
    $effect(() => () => base.dispose());
</script>

<T.Group>
    <T.Mesh position={[0, 0, -depth]}>
        <T.PlaneGeometry args={[width, height]} />
        <T.MeshStandardMaterial map={maps.back} color="#c9ced6" roughness={0.95} />
    </T.Mesh>
    <T.Mesh position={[0, -height / 2, -depth / 2]} rotation.x={-Math.PI / 2}>
        <T.PlaneGeometry args={[width, depth]} />
        <T.MeshStandardMaterial map={maps.floor} color="#eceef2" roughness={0.9} />
    </T.Mesh>
    <T.Mesh position={[0, height / 2, -depth / 2]} rotation.x={Math.PI / 2}>
        <T.PlaneGeometry args={[width, depth]} />
        <T.MeshStandardMaterial map={maps.ceiling} color="#b4bac4" roughness={0.95} />
    </T.Mesh>
    <T.Mesh position={[-width / 2, 0, -depth / 2]} rotation.y={Math.PI / 2}>
        <T.PlaneGeometry args={[depth, height]} />
        <T.MeshStandardMaterial map={maps.side} color="#d6dae1" roughness={0.95} />
    </T.Mesh>
    <T.Mesh position={[width / 2, 0, -depth / 2]} rotation.y={-Math.PI / 2}>
        <T.PlaneGeometry args={[depth, height]} />
        <T.MeshStandardMaterial map={maps.side} color="#d6dae1" roughness={0.95} />
    </T.Mesh>
</T.Group>
