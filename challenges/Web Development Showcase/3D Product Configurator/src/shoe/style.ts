import { type MeshPhysicalMaterial } from 'three';
import { FINISHES, type FinishId } from '../parts.ts';

/**
 * The detail map in the model is stored at half brightness, so a part's colour times two is the
 * colour it shows (see prepare-shoe.ts), and the model's roughness map is scaled by the finish.
 */
export const DETAIL_GAIN = 2;

export function styleMaterial(material: MeshPhysicalMaterial, colour: string, finish: FinishId): void {
    material.color.set(colour).multiplyScalar(DETAIL_GAIN);
    const f = FINISHES[finish];
    material.roughness = f.roughness;
    material.clearcoat = f.clearcoat;
    material.clearcoatRoughness = f.clearcoatRoughness;
}
