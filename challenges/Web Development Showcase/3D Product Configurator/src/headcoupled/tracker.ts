import { type Landmark } from './pose.ts';

/** One face's 478 landmarks per video frame, from MediaPipe's FaceLandmarker running in the page. */
export interface FaceTracker {
    readonly delegate: 'GPU' | 'CPU';
    /** Synchronous: it blocks the page for as long as the model takes (about 5-25 ms). */
    detect(video: HTMLVideoElement, timestampMs: number): Landmark[] | null;
    close(): void;
}

/**
 * Load the model and its WebAssembly runtime (13 MB, fetched only now, from this site: nothing comes
 * from a CDN), on the GPU if it will, else the CPU. `base` is where the page lives.
 */
export async function createTracker(base: string = document.baseURI): Promise<FaceTracker> {
    const { FaceLandmarker, FilesetResolver } = await import('@mediapipe/tasks-vision');
    const vision = await FilesetResolver.forVisionTasks(new URL('mediapipe/', base).href);
    const modelAssetPath = new URL('models/face_landmarker.task', base).href;
    const make = (delegate: 'GPU' | 'CPU') => FaceLandmarker.createFromOptions(vision, { baseOptions: { modelAssetPath, delegate }, runningMode: 'VIDEO', numFaces: 1 });

    let delegate: 'GPU' | 'CPU' = 'GPU';
    let landmarker: Awaited<ReturnType<typeof make>> | null = await make('GPU').catch(() => {
        delegate = 'CPU';
        return make('CPU');
    });
    return {
        get delegate() {
            return delegate;
        },
        detect(video, timestampMs) {
            if (!landmarker) return null; // the CPU model is still loading
            try {
                return landmarker.detectForVideo(video, timestampMs).faceLandmarks[0] ?? null;
            } catch (error) {
                if (delegate === 'CPU') throw error;
                // A GPU that creates fine and then fails on the first frame: carry on with the CPU.
                landmarker.close();
                landmarker = null;
                delegate = 'CPU';
                void make('CPU').then((l) => (landmarker = l));
                return null;
            }
        },
        close: () => landmarker?.close(),
    };
}
