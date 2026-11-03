import type { GarmentPhysicalProfile } from '../garment';
export type EvidenceConfidence = 'low' | 'medium' | 'high';
export type FitOutcome = 'too_small' | 'snug' | 'perfect' | 'relaxed' | 'too_large' | 'unknown';
export type FitZone =
  | 'shoulders'
  | 'chest'
  | 'waist'
  | 'hips'
  | 'thigh'
  | 'rise'
  | 'inseam'
  | 'sleeve'
  | 'torso_length'
  | 'shoe_length'
  | 'shoe_width';
export type ZoneOutcome = 'tight' | 'good' | 'loose' | 'short' | 'long' | 'unknown';
export interface FitEvidence {
  id: string;
  userId: string;
  productId?: string;
  productFamilyKey?: string;
  variantId?: string;
  brand?: string;
  category: string;
  subcategory?: string;
  sizeLabel?: string;
  cut?: string;
  materials?: string[];
  stretch?: string;
  source:
    'owned' | 'purchase' | 'try_on' | 'return' | 'manual' | 'body_profile' | 'reference_garment';
  preferredFit?: 'snug' | 'regular' | 'relaxed';
  overallFit: FitOutcome;
  zones?: Partial<Record<FitZone, ZoneOutcome>>;
  kept?: boolean;
  returned?: boolean;
  sold?: boolean;
  returnReason?: string;
  confidence: number;
  createdAt: string;
  sourceVersion?: string;
  reference?: boolean;
  physicalProfile?: GarmentPhysicalProfile;
}
export interface ReferenceGarment {
  evidenceId: string;
  productId?: string;
  brand: string;
  category: string;
  subcategory?: string;
  sizeLabel: string;
  productFamilyKey?: string;
  cut?: string;
  materials?: string[];
  stretch?: string;
  zones?: FitEvidence['zones'];
  physicalProfile?: GarmentPhysicalProfile;
  confidence: number;
}
export interface BrandSizePrior {
  brand: string;
  category: string;
  subcategory?: string;
  sizeLabel: string;
  productFamilyKey?: string;
  cut?: string;
  materialKey: string;
  stretch?: string;
  evidenceIds: string[];
  sources: FitEvidence['source'][];
  effectiveCount: number;
  positiveWeight: number;
  negativeWeight: number;
  zoneNotes: Partial<Record<FitZone, ZoneOutcome[]>>;
}
export interface FitProfile {
  version: 1;
  userId?: string;
  computedAt?: string;
  evidenceCount: number;
  lastUpdatedAt: string | null;
  fitPreferenceConfidence: EvidenceConfidence;
  preferredEaseByCategory: Record<string, 'snug' | 'regular' | 'relaxed'>;
  preferredEaseByZone: Partial<Record<FitZone, ZoneOutcome>>;
  brandCategorySizePriors: BrandSizePrior[];
  referenceGarments: ReferenceGarment[];
}
export interface SizeRecommendation {
  recommendedSize?: string;
  confidence: EvidenceConfidence;
  expectedFitIntent?: string;
  notes: string[];
  zoneNotes: string[];
  evidenceIds: string[];
  evidenceCount: number;
  provenance: string[];
  alternatives: Array<{ sizeLabel: string; support: number }>;
}
export interface PhysicalDataEvidence {
  source:
    | 'exact_chart'
    | 'merchant_variant'
    | 'stored_profile'
    | 'reference'
    | 'category_prior'
    | 'text'
    | 'unknown';
  confidence: EvidenceConfidence;
  profile?: GarmentPhysicalProfile;
  notes: string[];
}
