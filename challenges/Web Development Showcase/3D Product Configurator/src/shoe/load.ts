import { type Mesh, type MeshPhysicalMaterial } from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { PART_IDS, type PartId } from '../parts.ts';

export type ShoeParts = Record<PartId, Mesh & { material: MeshPhysicalMaterial }>;

/** Load the shoe: one mesh per part, each with its own material (the three textures are shared). */
export async function loadShoe(url: string): Promise<ShoeParts> {
    const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
    const gltf = await loader.loadAsync(url);
    const parts = {} as ShoeParts;
    for (const id of PART_IDS) {
        const mesh = gltf.scene.getObjectByName(id) as ShoeParts[PartId] | undefined;
        if (!mesh || !(mesh as Mesh).isMesh || !(mesh.material as MeshPhysicalMaterial).isMeshPhysicalMaterial) throw new Error(`the model has no part called ${id}`);
        parts[id] = mesh;
    }
    return parts;
}
