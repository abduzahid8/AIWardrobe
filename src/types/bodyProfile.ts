/**
 * Body profile + fit assessment types.
 *
 * This is the foundation of the body-fit try-on product (per
 * docs/AIWARDROBE_6_MONTH_BODY_FIT_VTON_PLAN.md, Month 1).
 *
 * Two new concepts on top of the existing avatarStore:
 *
 *   BodyProfile         — a user's calibrated digital body, with measurements
 *                         and privacy preferences. May be derived from manual
 *                         entry (MVP), SAM 3D Body photo analysis (Month 3),
 *                         or hybrid sources.
 *
 *   GarmentPhysicalProfile — per-garment physical measurements (chest, waist,
 *                            inseam, …) and fabric metadata (stretch, fit
 *                            intent, material). Drives the fit engine.
 *
 *   FitAssessment       — the engine's verdict: overall fit, per-zone notes,
 *                         confidence, and optional size recommendation.
 *
 * The shape is intentionally forward-compatible: Month 3 will add a
 * `mesh` field to BodyProfile (SAM 3D Body output) and a `measurementSourceHistory`
 * for versioned correction tracking. Month 5 will extend GarmentPhysicalProfile
 * with size-variant support.
 */

/** Where a measurement came from. */
export type BodyProfileSource =
  | 'manual'
  | 'apple_measure'
  | 'photo_sam_3d_body'
  | 'arkit_height'
  | 'hybrid';

/** How much we trust a measurement. */
export type Confidence = 'low' | 'medium' | 'high';

/** A single body measurement, in centimetres, with provenance. */
export interface BodyMeasurement {
  valueCm: number;
  confidence: Confidence;
  source: BodyProfileSource;
  /** When the user last edited this value. ISO-8601. */
  updatedAt?: string;
}

/** Body silhouette classifier used by mannequin3D.ts. */
export type BodyTypeId =
  | 'ectomorph'
  | 'average'
  | 'mesomorph'
  | 'endomorph'
  | 'hourglass'
  | 'pear';

/** Gender options for sizing. Mirrors auth.ts signup values. */
export type GenderOption = 'male' | 'female' | 'other' | 'prefer_not_to_say';

/** Lifecycle status of a body profile. */
export type BodyProfileStatus = 'draft' | 'analyzing' | 'ready' | 'failed';

/**
 * The user's calibrated digital body. A user can have multiple profiles
 * (e.g. one manual, one from a photo) but only one is `active` at a time
 * — that one drives the mannequin and fit engine.
 */
export interface BodyProfile {
  id: string;
  userId: string;
  name?: string; // user-given label, e.g. "Default", "After weight loss"
  status: BodyProfileStatus;
  isActive: boolean;

  // ── Identity ────────────────────────────────────────────────────────────
  gender?: GenderOption;
  height: BodyMeasurement; // required, high-confidence anchor
  weightKg?: number;
  bodyType?: BodyTypeId;

  // ── Measurements (all optional — fill in over time) ─────────────────────
  measurements: {
    shoulderWidth?: BodyMeasurement;
    chest?: BodyMeasurement;
    waist?: BodyMeasurement;
    hips?: BodyMeasurement;
    torsoLength?: BodyMeasurement;
    armLength?: BodyMeasurement;
    sleeveLength?: BodyMeasurement;
    inseam?: BodyMeasurement;
    thigh?: BodyMeasurement;
    calf?: BodyMeasurement;
    footLength?: BodyMeasurement;
  };

  // ── Mesh (populated by SAM 3D Body in Month 3) ──────────────────────────
  mesh?: {
    provider: 'sam_3d_body';
    meshUrl?: string;
    previewImageUrl?: string;
    rawOutputUrl?: string;
    version: string;
  };

  // ── Privacy ─────────────────────────────────────────────────────────────
  privacy: {
    retainSourcePhoto: boolean;
    retainMesh: boolean;
  };

  // ── Versioning (added in Month 4 for measurement history) ───────────────
  version: number;
  createdAt: string;
  updatedAt: string;
}

/** Helper: an empty draft body profile for the create flow. */
export function createEmptyBodyProfile(userId: string): BodyProfile {
  const now = new Date().toISOString();
  return {
    id: `bp_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    userId,
    status: 'draft',
    isActive: true,
    height: { valueCm: 175, confidence: 'medium', source: 'manual', updatedAt: now },
    weightKg: 70,
    bodyType: 'average',
    measurements: {},
    privacy: {
      retainSourcePhoto: false,
      retainMesh: true,
    },
    version: 1,
    createdAt: now,
    updatedAt: now,
  };
}
