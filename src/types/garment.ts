/**
 * Garment physical profile — drives the fit engine.
 *
 * A GarmentPhysicalProfile captures everything we know about a piece of
 * clothing that affects physical fit, beyond what a photo can show:
 * measurements, fabric stretch, fit intent, materials.
 *
 * A single product can have multiple size profiles (sizeVariant) — Month 5
 * will add the size-chart ingestion pipeline.
 *
 * Today (Month 1) we ship the shape + seed data so the fit engine has
 * something to work with. Month 5 will wire the admin/catalog ingestion.
 */

import { Confidence } from './bodyProfile';

/** How the garment is meant to fit the body, driven by design intent. */
export type FitIntent = 'compression' | 'slim' | 'regular' | 'relaxed' | 'oversized';

/** How much the fabric gives. 'high' means the garment stretches significantly. */
export type Stretch = 'none' | 'low' | 'medium' | 'high';

/** Category drives which zones the fit engine evaluates. */
export type GarmentCategory =
  | 'top'
  | 'shirt'
  | 'jacket'
  | 'coat'
  | 'pants'
  | 'jeans'
  | 'skirt'
  | 'dress'
  | 'shoes';

/** A single garment measurement, in centimetres. */
export interface GarmentMeasurement {
  valueCm: number;
  unitSource?: 'cm' | 'inch';
  confidence?: Confidence;
}

/**
 * Per-size physical profile. Most products have one profile per size label
 * (S/M/L/XL) or per numeric size (28/30/32).
 */
export interface GarmentPhysicalProfile {
  /** Unique garment + size combination ID. */
  id: string;
  /** Parent garment (product) ID. */
  garmentId: string;
  sizeLabel: string; // "S", "M", "L", "30", "32x32", etc.
  category: GarmentCategory;
  fitIntent?: FitIntent;
  stretch?: Stretch;
  material?: string[]; // ["cotton", "elastane"] — affects stretch defaults
  measurements: {
    chest?: GarmentMeasurement;
    waist?: GarmentMeasurement;
    hips?: GarmentMeasurement;
    shoulderWidth?: GarmentMeasurement;
    sleeveLength?: GarmentMeasurement;
    bodyLength?: GarmentMeasurement;
    inseam?: GarmentMeasurement;
    rise?: GarmentMeasurement;
    thigh?: GarmentMeasurement;
    hemOpening?: GarmentMeasurement;
    shoeLength?: GarmentMeasurement;
  };
  /** Where this data came from — helps the engine trust it. */
  source?: 'manufacturer_size_chart' | 'admin_manual' | 'user_self_measured' | 'unknown';
  /** Optional note shown to the user (e.g. "Runs small — size up"). */
  notes?: string;
}

/**
 * Seed data for known garments so the fit engine has something to work with
 * during MVP. Replace with real size charts in Month 5.
 *
 * Numbers are typical averages for a men's/unisex fit at 175 cm / 70 kg.
 * Sources: brand size charts + averaged public data.
 */
export const SEED_GARMENT_PHYSICAL_PROFILES: GarmentPhysicalProfile[] = [
  // ── Tops (chest, shoulder, sleeve) ────────────────────────────────────────
  {
    id: 'seed_tshirt_m',
    garmentId: 'tshirt_classic',
    sizeLabel: 'M',
    category: 'top',
    fitIntent: 'regular',
    stretch: 'low',
    material: ['cotton'],
    measurements: {
      chest: { valueCm: 100, confidence: 'medium' },
      shoulderWidth: { valueCm: 45, confidence: 'medium' },
      sleeveLength: { valueCm: 22, confidence: 'medium' },
      bodyLength: { valueCm: 70, confidence: 'medium' },
    },
    source: 'admin_manual',
  },
  {
    id: 'seed_tshirt_l',
    garmentId: 'tshirt_classic',
    sizeLabel: 'L',
    category: 'top',
    fitIntent: 'regular',
    stretch: 'low',
    material: ['cotton'],
    measurements: {
      chest: { valueCm: 108, confidence: 'medium' },
      shoulderWidth: { valueCm: 48, confidence: 'medium' },
      sleeveLength: { valueCm: 23, confidence: 'medium' },
      bodyLength: { valueCm: 72, confidence: 'medium' },
    },
    source: 'admin_manual',
  },
  {
    id: 'seed_oversized_hoodie_l',
    garmentId: 'hoodie_oversized',
    sizeLabel: 'L',
    category: 'top',
    fitIntent: 'oversized',
    stretch: 'medium',
    material: ['cotton', 'polyester'],
    measurements: {
      chest: { valueCm: 128, confidence: 'medium' },
      shoulderWidth: { valueCm: 58, confidence: 'medium' },
      sleeveLength: { valueCm: 65, confidence: 'medium' },
      bodyLength: { valueCm: 75, confidence: 'medium' },
    },
    source: 'admin_manual',
  },
  // ── Pants (waist, hips, thigh, inseam) ───────────────────────────────────
  {
    id: 'seed_jeans_32',
    garmentId: 'jeans_slim',
    sizeLabel: '32',
    category: 'jeans',
    fitIntent: 'slim',
    stretch: 'medium',
    material: ['denim', 'elastane'],
    measurements: {
      waist: { valueCm: 81, confidence: 'medium' },
      hips: { valueCm: 102, confidence: 'medium' },
      thigh: { valueCm: 58, confidence: 'medium' },
      inseam: { valueCm: 81, confidence: 'medium' },
      rise: { valueCm: 26, confidence: 'low' },
      hemOpening: { valueCm: 18, confidence: 'low' },
    },
    source: 'admin_manual',
  },
  {
    id: 'seed_chinos_32',
    garmentId: 'chinos_classic',
    sizeLabel: '32',
    category: 'pants',
    fitIntent: 'regular',
    stretch: 'low',
    material: ['cotton'],
    measurements: {
      waist: { valueCm: 84, confidence: 'medium' },
      hips: { valueCm: 104, confidence: 'medium' },
      thigh: { valueCm: 60, confidence: 'medium' },
      inseam: { valueCm: 82, confidence: 'medium' },
      rise: { valueCm: 27, confidence: 'low' },
      hemOpening: { valueCm: 20, confidence: 'low' },
    },
    source: 'admin_manual',
  },
  // ── Jacket (layer) ───────────────────────────────────────────────────────
  {
    id: 'seed_jacket_m',
    garmentId: 'jacket_bomber',
    sizeLabel: 'M',
    category: 'jacket',
    fitIntent: 'regular',
    stretch: 'low',
    material: ['polyester', 'cotton'],
    measurements: {
      chest: { valueCm: 112, confidence: 'medium' },
      shoulderWidth: { valueCm: 48, confidence: 'medium' },
      sleeveLength: { valueCm: 64, confidence: 'medium' },
      bodyLength: { valueCm: 67, confidence: 'medium' },
    },
    source: 'admin_manual',
  },
  // ── Shoes ────────────────────────────────────────────────────────────────
  {
    id: 'seed_shoes_eu42',
    garmentId: 'shoes_sneaker',
    sizeLabel: 'EU 42',
    category: 'shoes',
    fitIntent: 'regular',
    stretch: 'low',
    material: ['leather', 'rubber'],
    measurements: {
      shoeLength: { valueCm: 27.0, confidence: 'high' },
    },
    source: 'manufacturer_size_chart',
  },
];

/** Look up a seed physical profile by garmentId + sizeLabel. */
export function findSeedProfile(garmentId: string, sizeLabel: string): GarmentPhysicalProfile | undefined {
  return SEED_GARMENT_PHYSICAL_PROFILES.find(
    (p) => p.garmentId === garmentId && p.sizeLabel === sizeLabel,
  );
}
