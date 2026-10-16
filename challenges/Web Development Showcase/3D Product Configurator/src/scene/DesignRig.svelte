<script lang="ts">
    import { T, useTask, useThrelte } from '@threlte/core/webgpu';
    import { type PerspectiveCamera, Spherical, Vector3 } from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
    import { type Studio } from '../studio.svelte.ts';

    let { studio }: { studio: Studio } = $props();
    const { dom, invalidate, size } = useThrelte();
    const FOV = 28;
    let camera = $state.raw<PerspectiveCamera>();
    let controls = $state.raw<OrbitControls>();

    // The pointer (and touch) orbit and zoom through three's OrbitControls; the keyboard does the same through studio.orbit.
    $effect(() => {
        if (!camera) return;
        const c = camera;
        const o = new OrbitControls(c, dom);
        controls = o;
        o.enableDamping = true;
        o.enablePan = false;
        o.minDistance = 0.25;
        o.maxDistance = 1.1;
        o.maxPolarAngle = Math.PI * 0.49;
        o.target.set(0, 0.055, 0);
        o.update();
        const changed = () => invalidate();
        o.addEventListener('change', changed);
        studio.orbit = (azimuth, polar, zoom) => {
            const offset = new Vector3().subVectors(c.position, o.target);
            const s = new Spherical().setFromVector3(offset);
            s.theta += azimuth;
            s.phi = Math.min(o.maxPolarAngle, Math.max(0.15, s.phi + polar));
            s.radius = Math.min(o.maxDistance, Math.max(o.minDistance, s.radius * zoom));
            c.position.copy(o.target).add(offset.setFromSpherical(s));
            o.update();
            invalidate();
        };
        return () => {
            o.removeEventListener('change', changed);
            o.dispose();
            controls = undefined;
            studio.orbit = () => {};
        };
    });

    // On a narrow screen the shoe would run off the sides: when the shape of the view changes, stand back far enough for it to fit.
    $effect(() => {
        const { width, height } = $size;
        if (!camera || !controls || !width || !height) return;
        const horizontal = 2 * Math.atan(Math.tan((FOV * Math.PI) / 360) * (width / height));
        const fit = Math.min(controls.maxDistance, 0.19 / Math.tan(horizontal / 2));
        const offset = new Vector3().subVectors(camera.position, controls.target);
        if (offset.length() >= fit) return;
        camera.position.copy(controls.target).add(offset.setLength(fit));
        controls.update();
        invalidate();
    });

    // Damping keeps the view gliding for a moment after a drag ends.
    useTask(() => controls?.update(), { autoInvalidate: false });
</script>

<T.PerspectiveCamera makeDefault fov={FOV} near={0.02} far={20} position={[0.36, 0.2, 0.46]} bind:ref={camera} />
