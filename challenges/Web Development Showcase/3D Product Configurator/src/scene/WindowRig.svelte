<script lang="ts">
    import { T, useTask, useThrelte } from '@threlte/core/webgpu';
    import { OffAxisCamera } from '../headcoupled/camera.ts';
    import { canvasOnScreen, eyeInCanvas, placement } from '../headcoupled/frame.ts';
    import { type HeadSession } from '../headcoupled/session.svelte.ts';
    import { type ShoeParts } from '../shoe/load.ts';
    import { SHOE } from '../shoe/dimensions.ts';
    import { type Studio } from '../studio.svelte.ts';
    import Backdrop from './Backdrop.svelte';
    import Room from './Room.svelte';
    import ShoeModel from './ShoeModel.svelte';

    let { studio, head, parts, base }: { studio: Studio; head: HeadSession; parts: ShoeParts; base: string } = $props();
    const { dom, renderer, invalidate } = useThrelte();

    const camera = new OffAxisCamera();
    camera.near = 0.01;
    camera.far = 20;

    /** How far back the room goes, and where the shoe stands in it (the middle of the shoe, behind the glass). */
    const DEPTH = 0.9;
    const SHOE_Z = -0.09;
    const SHOE_TURN = -0.55;
    /** The window, in metres: what the room is built to fit, so it only changes when the canvas does. */
    let room = $state({ width: 0.31, height: 0.17 });
    let fit = $state(1);

    // Without a camera the pointer stands in for the head: where it is in the window is where the eye is.
    let pointer = { x: 0, y: 0 };
    let smooth = { x: 0, y: 0, z: 0.55 };
    $effect(() => {
        const el = dom;
        const move = (e: PointerEvent) => {
            const r = el.getBoundingClientRect();
            pointer = { x: ((e.clientX - r.left) / r.width) * 2 - 1, y: -(((e.clientY - r.top) / r.height) * 2 - 1) };
        };
        const leave = () => (pointer = { x: 0, y: 0 });
        el.addEventListener('pointermove', move);
        el.addEventListener('pointerleave', leave);
        return () => {
            el.removeEventListener('pointermove', move);
            el.removeEventListener('pointerleave', leave);
        };
    });

    useTask((delta) => {
        const canvas = renderer.domElement as HTMLCanvasElement;
        const origin = screen as Screen & { availLeft?: number; availTop?: number };
        const p = placement({ width: screen.width, height: screen.height, diagonal: head.settings.diagonal }, canvasOnScreen(canvas.getBoundingClientRect(), window, origin.availLeft ?? 0, origin.availTop ?? 0));
        if (Math.abs(p.size.width - room.width) > 1e-4 || Math.abs(p.size.height - room.height) > 1e-4) room = { width: p.size.width, height: p.size.height };

        let eye: { x: number; y: number; z: number };
        let source: 'camera' | 'pointer' | 'override';
        if (studio.eyeOverride) {
            eye = studio.eyeOverride;
            source = 'override';
        } else if (head.status === 'tracking' || head.status === 'no-face') {
            eye = eyeInCanvas(head.eye, p.offset);
            source = 'camera';
        } else {
            const k = 1 - Math.exp(-delta / 0.1);
            smooth = { x: smooth.x + (pointer.x * p.size.width * 0.8 - smooth.x) * k, y: smooth.y + (pointer.y * p.size.height * 0.8 - smooth.y) * k, z: 0.55 };
            eye = smooth;
            source = 'pointer';
        }
        camera.place(eye, p.size);
        studio.seen = { eye: { ...eye }, width: p.size.width, height: p.size.height, source };

        // Life size when it fits; on a small screen, as large as leaves the walls visible.
        const scale = Math.min(1, (0.7 * p.size.width) / SHOE.length, (0.6 * p.size.height) / SHOE.height);
        if (Math.abs(scale - fit) > 1e-3) {
            fit = scale;
            studio.shoeScale = scale;
        }
        invalidate();
    });
</script>

<T is={camera} makeDefault manual />
<T.DirectionalLight position={[0.3, 0.6, 0.5]} intensity={1.6} />
<Room width={room.width} height={room.height} depth={DEPTH} />
<Backdrop url={new URL('models/shadow.png', base).href} strength={0.5} position={[0, -room.height / 2, SHOE_Z]} rotationY={SHOE_TURN} scale={fit} />
<ShoeModel {parts} {studio} position={[0, -room.height / 2, SHOE_Z]} rotationY={SHOE_TURN} scale={fit} />
