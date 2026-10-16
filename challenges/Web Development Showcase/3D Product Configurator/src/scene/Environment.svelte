<script lang="ts">
    import { useThrelte } from '@threlte/core/webgpu';
    import { EquirectangularReflectionMapping } from 'three';
    import { HDRLoader } from 'three/addons/loaders/HDRLoader.js';

    let { url, intensity = 1, onerror }: { url: string; intensity?: number; onerror?: (message: string) => void } = $props();
    const { scene, invalidate } = useThrelte();

    // A photographed studio (Poly Haven, CC0) lights and reflects in the shoe; the page shows its own backdrop.
    $effect(() => {
        let gone = false;
        new HDRLoader()
            .loadAsync(url)
            .then((texture) => {
                if (gone) return texture.dispose();
                texture.mapping = EquirectangularReflectionMapping;
                scene.environment = texture;
                invalidate();
            })
            .catch((error: unknown) => onerror?.(`The lighting could not load: ${error instanceof Error ? error.message : String(error)}`));
        return () => {
            gone = true;
            scene.environment = null;
        };
    });

    $effect(() => {
        scene.environmentIntensity = intensity;
        invalidate();
    });
</script>
