/**
 * visualOS.ts — Multimodal stylist domain types (Phase 4).
 *
 * Vision is evidence, not truth: every inference carries confidence,
 * provenance, model version and user-correction state. Structured
 * validators stay authoritative.
 */

export type VisualAssetKind =
  | 'original_upload'
  | 'cropped_garment'
  | 'background_removed'
  | 'thumbnail'
  | 'normalized'
  | 'tryon_result'
  | 'inspiration'
  | 'outfit_photo'
  | 'merchant_image';

export interface VisualAsset {
  id: string;
  kind: VisualAssetKind;
  originalAssetId?: string | null;
  transformationChain: string[];
  localUri?: string | null;
  remoteUri?: string | null;
  contentHash?: string | null;
  width?: number | null;
  height?: number | null;
  mimeType?: string | null;
  createdAt: string;
  source: 'user' | 'merchant' | 'generated' | 'import';
  retentionClass: 'ephemeral' | 'session' | 'durable';
  userOwned: boolean;
}

export type VisualEntityType =
  | 'wardrobe_item' | 'catalog_product' | 'outfit'
  | 'inspiration' | 'mirror_photo' | 'receipt_or_page';

export type VisualAttributeType =
  | 'garment_type' | 'category' | 'color' | 'pattern' | 'material'
  | 'silhouette' | 'fit_intent' | 'formality' | 'style_tags'
  | 'brand' | 'condition' | 'layering' | 'outfit_role' | 'visual_similarity';

export type VisualSource =
  | 'user' | 'metadata' | 'deterministic_image_analysis'
  | 'local_model' | 'remote_model' | 'merchant';

export interface VisualEvidence {
  id: string;
  userId: string;
  sourceAssetId: string;
  entityType: VisualEntityType;
  entityId?: string | null;
  attributeType: VisualAttributeType;
  value: string;
  confidence: number;
  source: VisualSource;
  modelVersion?: string | null;
  createdAt: string;
  supersededBy?: string | null;
  userConfirmed: boolean;
  userRejected: boolean;
}

export type ImageUsability = 'usable' | 'degraded' | 'unusable';
export interface QualityGateResult {
  usability: ImageUsability;
  reasons: string[];
  recommendedCorrection?: string | null;
  analyzedAt: string;
}

export interface GarmentRegion {
  id: string;
  parentAssetId: string;
  bbox: { x: number; y: number; w: number; h: number };
  confidence: number;
  proposedType?: string | null;
  cropAssetId?: string | null;
  manual: boolean;
}

export interface GarmentVisualParse {
  assetId: string;
  garmentType: string;
  category: string;
  dominantColors: string[];
  secondaryColors: string[];
  pattern: string;
  silhouette: string;
  materialClass: string;
  formality: number;
  styleTags: string[];
  visibleDetails: string[];
  conditionHint: string;
  confidencePerField: Record<string, number>;
  provenance: string;
  parsedAt: string;
}

export interface OutfitVisualParse {
  assetId: string;
  upperBody?: string | null;
  lowerBody?: string | null;
  outerwear?: string | null;
  shoes?: string | null;
  accessories: string[];
  dominantPalette: string[];
  contrast: 'low' | 'medium' | 'high' | 'unknown';
  silhouette: string;
  layering: number;
  formality: number;
  styleDescriptors: string[];
  balanceNotes: string[];
  confidence: number;
  provenance: string;
  parsedAt: string;
}

export type FingerprintVerdict =
  | 'SAME_ITEM_HIGH_CONFIDENCE' | 'SAME_ITEM_POSSIBLE'
  | 'SAME_FAMILY' | 'VISUALLY_SIMILAR' | 'DISTINCT';

export interface FingerprintResult {
  verdict: FingerprintVerdict;
  score: number;
  evidence: string[];
  requiresConfirmation: boolean;
}

export type ScanQueueStatus =
  | 'queued' | 'quality_check' | 'awaiting_confirm'
  | 'duplicate_review' | 'saved' | 'skipped' | 'failed';

export interface ScanQueueItem {
  id: string;
  userId: string;
  assetId: string;
  status: ScanQueueStatus;
  proposedCategory?: string | null;
  proposedColor?: string | null;
  proposedType?: string | null;
  duplicateOfItemId?: string | null;
  attempts: number;
  updatedAt: string;
}

export interface CompatibilityDimension {
  key: string;
  score: number;
  note: string;
  confidence: 'low' | 'medium' | 'high';
}

export interface OutfitCompatibility {
  outfitId?: string | null;
  dimensions: CompatibilityDimension[];
  strengths: string[];
  tensions: string[];
  hardConflicts: string[];
  improvements: string[];
  confidence: 'low' | 'medium' | 'high';
  provenance: string[];
}

export interface OutfitDiagnosis {
  issue: string;
  detail: string;
  actionableFix: string;
  confidence: 'low' | 'medium' | 'high';
}

export interface OutfitTransformation {
  beforeItemIds: string[];
  afterItemIds: string[];
  swappedOut: string[];
  swappedIn: string[];
  rationale: string[];
  preservedAnchors: string[];
  confidence: 'low' | 'medium' | 'high';
}

export interface ReferenceLook {
  id: string;
  userId: string;
  assetId: string;
  sourceLabel: string;
  styleRepresentation: Record<string, string[]>;
  userRating?: number | null;
  savedAt: string;
  contexts: string[];
  recreatedOutfitIds: string[];
  matchConfidence?: 'low' | 'medium' | 'high' | null;
}

export interface RecreateLookResult {
  referenceId: string;
  closestWardrobeOutfit: string[] | null;
  bestPracticalOutfit: string[] | null;
  coverage: number;
  missingRoles: string[];
  missingPieceSuggestion?: string | null;
  explanation: string[];
  confidence: 'low' | 'medium' | 'high';
}

export type TwinCondition =
  | 'excellent' | 'good' | 'worn' | 'needs_attention'
  | 'repair' | 'replace_candidate' | 'unknown';

export interface WardrobeTwinItem {
  itemId: string;
  identity: Record<string, string>;
  visualFeatures: Record<string, string[]>;
  condition: TwinCondition;
  fitSummary?: string | null;
  styleSummary?: string | null;
  lifecycleSummary?: string | null;
  usageSummary?: string | null;
  companions: string[];
  substitutes: string[];
  provenance: Record<string, string>;
}

export type LifecycleDecision =
  | 'KEEP' | 'REPAIR' | 'REPLACE_WHEN_NEEDED'
  | 'SELL' | 'DONATE' | 'ARCHIVE' | 'INSUFFICIENT_EVIDENCE';

export interface LifecycleDecisionResult {
  itemId: string;
  decision: LifecycleDecision;
  evidence: string[];
  confidence: 'low' | 'medium' | 'high';
}

export interface OutfitBoard {
  id: string;
  userId: string;
  itemIds: string[];
  lockedItemIds: string[];
  alternativesRequestedFor?: string | null;
  versions: Array<{ label: string; itemIds: string[]; createdAt: string }>;
  diagnostics?: OutfitDiagnosis[] | null;
  updatedAt: string;
}

export interface TryOnRequest {
  id: string;
  personAssetId: string;
  garmentVariantId: string;
  garmentImageAssetId: string;
  category: string;
  preservationConstraints: string[];
  provider: string;
  qualityPreset: 'draft' | 'standard' | 'high';
}

export interface TryOnResult {
  requestId: string;
  outputAssetId?: string | null;
  provider: string;
  modelVersion?: string | null;
  latencyMs?: number | null;
  status: 'succeeded' | 'failed' | 'degraded' | 'unsupported';
  limitations: string[];
  provenance: string;
  /** Synthetic renders must never feed FitEvidence as real fit proof. */
  synthetic: boolean;
}

export type StylistToolName =
  | 'wardrobe.query' | 'outfit.generate' | 'outfit.validate'
  | 'outfit.diagnose' | 'outfit.transform' | 'style.getProfile'
  | 'fit.evaluate' | 'shopping.buyCheck' | 'shopping.findBetter'
  | 'shopping.findMissingPiece' | 'shopping.maxWardrobeGain'
  | 'price.evaluate' | 'visual.parseGarment' | 'visual.parseOutfit'
  | 'visual.recreateLook' | 'planner.getEvents'
  | 'planner.getWeatherContext' | 'planner.getWardrobeGoals';

export interface StylistPlan {
  id: string;
  intent: string;
  requestedConstraints: string[];
  toolsSelected: StylistToolName[];
  candidateCounts: Record<string, number>;
  validationStages: string[];
  fallbackPath?: string | null;
  evidenceSummary: string[];
  createdAt: string;
}

export interface StylistResult {
  planId: string;
  outfitItemIds: string[] | null;
  backupItemIds?: string[] | null;
  explanation: string[];
  evidence: string[];
  confidence: 'low' | 'medium' | 'high';
  fallbacksUsed: string[];
}

export type VisualFlag =
  | 'closet_scan' | 'visual_duplicates' | 'inspiration_recreate'
  | 'outfit_diagnostics' | 'visual_search' | 'stylist_orchestration';

export interface InferenceCacheEntry {
  key: string;
  task: string;
  modelVersion: string;
  schemaVersion: string;
  result: unknown;
  cachedAt: string;
}

export interface SanitizedTelemetry {
  task: string;
  provider: string;
  latencyMs?: number | null;
  cacheHit: boolean;
  fallbackUsed?: string | null;
  confidence?: number | null;
  failureClass?: string | null;
  day: string;
}
