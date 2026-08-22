/**
 * JS port of the seed garment physical profiles (api/services/garmentSeed.js).
 * Mirrors src/types/garment.ts SEED_GARMENT_PHYSICAL_PROFILES. See that file
 * for sourcing notes. KEEP IN SYNC.
 */

export const SEED_GARMENT_PHYSICAL_PROFILES = [
  { id: 'seed_tshirt_m', garmentId: 'tshirt_classic', sizeLabel: 'M', category: 'top', fitIntent: 'regular', stretch: 'low', material: ['cotton'],
    measurements: { chest: { valueCm: 100, confidence: 'medium' }, shoulderWidth: { valueCm: 45, confidence: 'medium' }, sleeveLength: { valueCm: 22, confidence: 'medium' }, bodyLength: { valueCm: 70, confidence: 'medium' } }, source: 'admin_manual' },
  { id: 'seed_tshirt_l', garmentId: 'tshirt_classic', sizeLabel: 'L', category: 'top', fitIntent: 'regular', stretch: 'low', material: ['cotton'],
    measurements: { chest: { valueCm: 108, confidence: 'medium' }, shoulderWidth: { valueCm: 48, confidence: 'medium' }, sleeveLength: { valueCm: 23, confidence: 'medium' }, bodyLength: { valueCm: 72, confidence: 'medium' } }, source: 'admin_manual' },
  { id: 'seed_oversized_hoodie_l', garmentId: 'hoodie_oversized', sizeLabel: 'L', category: 'top', fitIntent: 'oversized', stretch: 'medium', material: ['cotton', 'polyester'],
    measurements: { chest: { valueCm: 128, confidence: 'medium' }, shoulderWidth: { valueCm: 58, confidence: 'medium' }, sleeveLength: { valueCm: 65, confidence: 'medium' }, bodyLength: { valueCm: 75, confidence: 'medium' } }, source: 'admin_manual' },
  { id: 'seed_jeans_32', garmentId: 'jeans_slim', sizeLabel: '32', category: 'jeans', fitIntent: 'slim', stretch: 'medium', material: ['denim', 'elastane'],
    measurements: { waist: { valueCm: 81, confidence: 'medium' }, hips: { valueCm: 102, confidence: 'medium' }, thigh: { valueCm: 58, confidence: 'medium' }, inseam: { valueCm: 81, confidence: 'medium' }, rise: { valueCm: 26, confidence: 'low' }, hemOpening: { valueCm: 18, confidence: 'low' } }, source: 'admin_manual' },
  { id: 'seed_chinos_32', garmentId: 'chinos_classic', sizeLabel: '32', category: 'pants', fitIntent: 'regular', stretch: 'low', material: ['cotton'],
    measurements: { waist: { valueCm: 84, confidence: 'medium' }, hips: { valueCm: 104, confidence: 'medium' }, thigh: { valueCm: 60, confidence: 'medium' }, inseam: { valueCm: 82, confidence: 'medium' }, rise: { valueCm: 27, confidence: 'low' }, hemOpening: { valueCm: 20, confidence: 'low' } }, source: 'admin_manual' },
  { id: 'seed_jacket_m', garmentId: 'jacket_bomber', sizeLabel: 'M', category: 'jacket', fitIntent: 'regular', stretch: 'low', material: ['polyester', 'cotton'],
    measurements: { chest: { valueCm: 112, confidence: 'medium' }, shoulderWidth: { valueCm: 48, confidence: 'medium' }, sleeveLength: { valueCm: 64, confidence: 'medium' }, bodyLength: { valueCm: 67, confidence: 'medium' } }, source: 'admin_manual' },
  { id: 'seed_shoes_eu42', garmentId: 'shoes_sneaker', sizeLabel: 'EU 42', category: 'shoes', fitIntent: 'regular', stretch: 'low', material: ['leather', 'rubber'],
    measurements: { shoeLength: { valueCm: 27.0, confidence: 'high' } }, source: 'manufacturer_size_chart' },
];

export function findSeedProfile(garmentId, sizeLabel) {
  return SEED_GARMENT_PHYSICAL_PROFILES.find(
    (p) => p.garmentId === garmentId && p.sizeLabel === sizeLabel,
  );
}
